// -*- c++ -*-
#include <moderngpu/kernel_mergesort.hxx>
#include <moderngpu/kernel_sortedsearch.hxx>
#include <moderngpu/transform.hxx>
#include <moderngpu/kernel_reduce.hxx>
#include <moderngpu/kernel_scan.hxx>

#include <util/random-numbers.hxx>

using num_t = float;

template<typename system_t>
struct ResamplingDMC {
  using walker_state_t = typename system_t::walker_state_t;
  using parameter_t = typename system_t::parameter_t;

  mgpu::mem_t<walker_state_t> old_walker_state;
  mgpu::mem_t<walker_state_t> new_walker_state;
  mgpu::mem_t<num_t> weights, probabilities, weights_prefix_sum;
  
  mgpu::context_t & context;
  uint seed = 10;
  float dt = 0.01;
  float sqrt_dt = sqrt(dt);
  uint num_walkers;
  uint iter = 0;
  
  ResamplingDMC(int num_walkers, mgpu::context_t & context) :
    num_walkers(num_walkers),
    context(context),
    old_walker_state(num_walkers, context),
    new_walker_state(num_walkers, context),
    weights(num_walkers, context),
    probabilities(num_walkers, context),
    weights_prefix_sum(num_walkers, context)
  {}

  void initialize() {
    auto old_walker_state_data = old_walker_state.data();

    /* Explicitly dereference this-pointer to avoid closing over it in
     * the lambda below. */
    auto seed = this->seed;
    
    // Initialize walker positions to random gaussians of std 1
    mgpu::transform(
      [=]MGPU_DEVICE(uint index) {
        auto randoms =
          gpu_random::gaussians<system_t::walker_dimension>(uint4{index, 0, 0, 0},
                                                            uint2{seed, 0});
        old_walker_state_data[index] = randoms;
      }, num_walkers, context);
  }

  float step(parameter_t guide_parameters) {
    auto local_iter = this->iter, seed = this->seed;
    auto sqrt_dt = this->sqrt_dt, dt = this->dt;
    auto num_walkers = this->num_walkers;
    
    auto old_walker_state_data = old_walker_state.data();
    mgpu::mem_t<math::vector_t<2,num_t>> energy_estimate_device(1, context);
  
    /* Perform diffusion and compute the weight for every walker. */
    auto weights_data = weights.data();
    mgpu::transform_reduce(
      [=]MGPU_DEVICE(uint index) {
        auto walker_state = old_walker_state_data[index];          

        auto diffusion_randoms =
          gpu_random::gaussians<system_t::walker_dimension>(uint4{index, local_iter, 0, 0},
                                                            uint2{seed, 1});
        
        /* Energy evaluation: evaluate the Hamiltonian at each
           walker's position. */
        auto energy_before = system_t::local_energy(walker_state, guide_parameters);

        auto drift = system_t::drift_velocity(walker_state, guide_parameters);
        /* Diffusion: add a random gaussian of stddev dt to each
           walker's position;
           Drift: add dt*drift velocity from guide wavefunction */
        walker_state += sqrt_dt*diffusion_randoms + dt*drift;
        auto energy_after = system_t::local_energy(walker_state, guide_parameters);

        old_walker_state_data[index] = walker_state;
        
        num_t weight = exp(-dt * (0.5*(energy_before + energy_after)));
        weights_data[index] = weight;
        return math::vector_t<2,num_t>{weight, weight * energy_after};
      },
      num_walkers,
      energy_estimate_device.data(),
      mgpu::plus_t<math::vector_t<2,num_t>>(),
      context);

    auto energy_estimate_host = mgpu::from_mem(energy_estimate_device)[0];
    auto total_weight = energy_estimate_host[0];
    auto weighted_local_energy = energy_estimate_host[1];
    
    assert(!std::isnan(total_weight));
    assert(std::isfinite(total_weight));
    assert(!std::isnan(weighted_local_energy));
    assert(std::isfinite(weighted_local_energy));
    assert(total_weight != 0);
    
    auto energy_estimate = weighted_local_energy / total_weight;
    //printf("%f %d\n", energy_estimate, num_walkers);    
    assert(!std::isnan(energy_estimate));
    assert(std::isfinite(energy_estimate));

    /* Compute the prefix-sum for weights. */
    mgpu::mem_t<num_t> should_be_one(1, context);
    mgpu::transform_scan<num_t,mgpu::scan_type_t::scan_type_inc>(
      [=]MGPU_DEVICE(uint index) {
        return weights_data[index]/total_weight;
      },
      num_walkers,
      weights_prefix_sum.data(),
      mgpu::plus_t<num_t>(),
      should_be_one.data(),
      context);

    double one = mgpu::from_mem(should_be_one)[0];
    assert(abs(one-1.0) < 1e-6);

    /* Generate sorted probabilities that will sample from the
     * previous generation. */
    auto probabilities_data = probabilities.data();
    mgpu::transform(
      [=]MGPU_DEVICE(uint index) {
        num_t jitter = gpu_random::uniforms<1>(uint4{index, local_iter, 0, 0}, uint2{seed, 2})[0];
        num_t random = num_t(index)/(num_walkers+1) + jitter/num_walkers;
        
        probabilities_data[index] = random;
      }, num_walkers, context);

    /* Don't need to explicitly sort them since we generated a set of
       sorted probabilities. */
    //mgpu::mergesort(probabilities.data(), num_walkers, mgpu::less_t<num_t>(), context);

    /* Binary-search each probability into the prefix-summed array of
     * weights to find the walker corresponding to that probability.
     * Copy over that walker to its place in the new generation. This
     * is accomplished with a single merge, since both sets are
     * already sorted. */
    auto new_walker_state_data = new_walker_state.data();
    auto weights_prefix_sum_data = weights_prefix_sum.data();
    mgpu::sorted_search<mgpu::bounds_lower>(
      probabilities.data(), num_walkers,
      weights_prefix_sum.data(), num_walkers,
      mgpu::make_store_iterator<int>([=]MGPU_DEVICE(uint parent_index, uint child_index) {
          if(parent_index >= num_walkers) {
            parent_index = num_walkers-1;
          }

          new_walker_state_data[child_index] = old_walker_state_data[parent_index];
        }),
      mgpu::less_t<num_t>(),
      context);

    old_walker_state.swap(new_walker_state);

    iter++;
    return energy_estimate;
  }
};
