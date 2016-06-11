#include <moderngpu/transform.hxx>
#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/kernel_load_balance.hxx>
#include <moderngpu/kernel_reduce.hxx>

#include <util/random-numbers.hxx>

template<typename system_t>
struct TraditionalDMC {
  using walker_state_t = typename system_t::walker_state_t;
  using parameter_t = typename system_t::parameter_t;
  
  mgpu::mem_t<walker_state_t> old_walker_state;
  mgpu::context_t & context;
  uint seed = 10;
  float dt = 0.01;
  float damping_alpha = 0.1;
  float sqrt_dt = sqrt(dt);
  uint num_walkers;
  uint iter = 0;
  float target_energy = 0.0;
  int target_num_walkers;
  
  TraditionalDMC(int target_num_walkers, mgpu::context_t & context) :
    num_walkers(target_num_walkers),
    target_num_walkers(target_num_walkers),
    context(context),
    old_walker_state(target_num_walkers, context)
  {}

  void initialize() {
    auto old_walker_state_data = old_walker_state.data();

    /* Explicitly dereference this-pointer to avoid closing over it in
     * the lambda below. */
    auto seed = this->seed;
    
    // Initialize all walker positions to random gaussians of std 1
    mgpu::transform(
      [=]MGPU_DEVICE(uint index) {
        auto randoms = gpu_random::uniforms<system_t::walker_dimension>(uint4{index, 0, 0, 0},
                                                                        uint2{seed, 0});
        mgpu::iterate<system_t::walker_dimension>([&](int dimension_index) {
            old_walker_state_data[index][dimension_index] = randoms[dimension_index];
          });
      }, num_walkers, context);
  }

  float step(parameter_t guide_parameters) {
    auto old_walker_state_data = old_walker_state.data();
    mgpu::mem_t<int> children(num_walkers, context);
    int* children_data = children.data();

    auto local_iter = this->iter, seed = this->seed;
    auto sqrt_dt = this->sqrt_dt, dt = this->dt, local_target_energy = this->target_energy;
    
    mgpu::mem_t<float2> energy_estimate_device(1, context);
    
    mgpu::transform_reduce(
      [=]MGPU_DEVICE(uint index) {
        auto walker_state = old_walker_state_data[index];          

        auto diffusion_randoms =
          gpu_random::gaussians<system_t::walker_dimension>(uint4{index, local_iter, 0, 0},
                                                            uint2{seed, 1});
        
        // Energy evaluation: evaluate the Hamiltonian at each walker's
        // position.
        auto energy_before = system_t::local_energy(walker_state, guide_parameters);

        auto drift = system_t::drift_velocity(walker_state, guide_parameters);
        // Diffusion: add a random gaussian of stddev dt to each walker's position;
        // Drift: add dt*drift velocity from guide wavefunction
        walker_state += sqrt_dt*diffusion_randoms + dt*drift;
        
        auto energy_after = system_t::local_energy(walker_state, guide_parameters);

        old_walker_state_data[index] = walker_state;
        // Birth-death I: calculate the number of copies of each walker in the
        // next generation

        // We will also estimate the average energy at this point since it
        // uses the branching-factor used to compute birth and death.
        float branching_factor = exp(-dt * (0.5*(energy_before + energy_after) - local_target_energy));

        auto branching_random = gpu_random::uniforms<1>(uint4{index, local_iter, 0, 0}, uint2{seed, 2})[0];

        children_data[index] = int(branching_factor + branching_random);

        return float2{branching_factor, branching_factor * energy_after};
        
      }, num_walkers,
      energy_estimate_device.data(),
      plus_float2_t(),
      context);

    float2 energy_estimate_host = mgpu::from_mem(energy_estimate_device)[0];
    assert(!std::isnan(energy_estimate_host.x));
    assert(std::isfinite(energy_estimate_host.x));
    assert(!std::isnan(energy_estimate_host.y));
    assert(std::isfinite(energy_estimate_host.y));
    assert(energy_estimate_host.x != 0);
    
    float energy_estimate = energy_estimate_host.y / energy_estimate_host.x;
//    printf("%f %d %f\n", energy_estimate, num_walkers, target_energy);    
    assert(!std::isnan(energy_estimate));
    assert(std::isfinite(energy_estimate));
    

    // Birth-death II: compute a prefix-sum of the number-of-copies for
    // each walker
    mgpu::mem_t<int> children_offsets(num_walkers, context);
    mgpu::mem_t<int> total_children(1, context);
    mgpu::scan(children.data(), num_walkers, children_offsets.data(),
               mgpu::plus_t<int>(), total_children.data(), context);

    // Birth-death III: Now create children-number of copies of each
    // walker into a second array.
    int num_children = mgpu::from_mem(total_children)[0];
    assert(num_children > 0);

    // Fill new_walker_state with the value of the parent walker.
    mgpu::mem_t<walker_state_t> new_walker_state(num_children, context);
    auto new_state = new_walker_state.data();

    mgpu::transform_lbs(
      [=]MGPU_DEVICE(int index, int parent, int sibling,
                     mgpu::tuple<walker_state_t> desc) {
        new_state[index] = mgpu::get<0>(desc);
      }, num_children, children_offsets.data(), num_walkers, 
      mgpu::make_tuple(old_walker_state.data()),
      context
      );

    old_walker_state.swap(new_walker_state);
    num_walkers = num_children;
    
    if(iter % 100 == 0)
      target_energy = energy_estimate;
    else
      target_energy += damping_alpha * (log(target_num_walkers) - log(num_walkers));

    iter++;
    return energy_estimate;
  }
};
