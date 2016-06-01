// -*- c++ -*-
#include <moderngpu/transform.hxx>
#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/kernel_load_balance.hxx>
#include <moderngpu/kernel_reduce.hxx>

#include <curand_kernel.h>
#include <curand_normal.h>

const int walker_dimension = 3;

struct alignas(8) walker_state_t {
  float pos[walker_dimension];
};

MGPU_DEVICE float harmonic_oscillator_hamiltonian(walker_state_t state) {
  float xx;
  for(int ii = 0; ii < walker_dimension; ++ii)
    xx += state.pos[ii]*state.pos[ii];
  return xx/2;
}

struct plus_float2_t : public std::binary_function<float2, float2, float2> {
  MGPU_HOST_DEVICE float2 operator()(float2 a, float2 b) const {
    return float2{a.x + b.x, a.y + b.y};
  }
};

struct array_float_4_t {
  float values[4];
};

MGPU_DEVICE array_float_4_t random_gaussians(uint4 counter, uint2 key) {
  uint4 result = curand_Philox4x32_10(counter, key);
  float2 hi = _curand_box_muller(result.x, result.y);
  float2 lo = _curand_box_muller(result.z, result.w);
  array_float_4_t answer;
  answer.values[0] = hi.x; answer.values[1] = hi.y; answer.values[2] = lo.x; answer.values[3] = lo.y;
  return answer;
}

int main(int argc, char** argv) {
  assert(argc == 3);
  uint niters = atoi(argv[1]);
  int target_num_walkers = atoi(argv[2]);
  
  uint seed = 10;
  float dt = 0.05;
  float damping_alpha = 0.1;

  float sqrt_dt = sqrt(dt);  
  int num_walkers = target_num_walkers;
  float target_energy = 1.0;

  mgpu::standard_context_t context;
  
  mgpu::mem_t<walker_state_t> old_walker_state(num_walkers, context);
  auto old_walker_state_data = old_walker_state.data();
  
  // Initialize all walker positions to random gaussians of std 1
  mgpu::transform(
    [=]MGPU_DEVICE(uint index) {
      auto randoms = random_gaussians(uint4{index, 0, 0, 0}, uint2{seed, 1});
      mgpu::iterate<walker_dimension>([&](int dimension_index) {
          old_walker_state_data[index].pos[dimension_index] = randoms.values[dimension_index];
        });
    }, num_walkers, context);
    
  for(uint iter = 0; iter < niters; ++iter) {
    old_walker_state_data = old_walker_state.data();
    mgpu::mem_t<int> children(num_walkers, context);
    int* children_data = children.data();
    
    mgpu::mem_t<float2> energy_estimate_device(1, context);
    
    mgpu::transform_reduce(
      [=]MGPU_DEVICE(uint index) {
        auto walker_state = old_walker_state_data[index];          
        
        auto randoms = random_gaussians(uint4{index, iter, 0, 0}, uint2{seed, 0});
        
        // Energy evaluation: evaluate the Hamiltonian at each walker's
        // position.
        auto energy_before = harmonic_oscillator_hamiltonian(walker_state);

        // Diffusion: add a random gaussian of stddev dt to each walker's position
        mgpu::iterate<walker_dimension>([&](int dimension_index) {
            walker_state.pos[dimension_index] += sqrt_dt * randoms.values[dimension_index];
          });
        auto energy_after = harmonic_oscillator_hamiltonian(walker_state);

        old_walker_state_data[index] = walker_state;
        // Birth-death I: calculate the number of copies of each walker in the
        // next generation

        // We will also estimate the average energy at this point since it
        // uses the branching-factor used to compute birth and death.
        float branching_factor = exp(-dt * (0.5*(energy_before + energy_after) - target_energy));
        uint4 rand_result = curand_Philox4x32_10(uint4{index, iter, 1, 0}, uint2{seed, 0});
        float uniform_float = _curand_uniform(rand_result.x);

        children_data[index] = int(branching_factor + uniform_float);

        return float2{branching_factor, branching_factor * energy_after};        
        
      }, num_walkers,
      energy_estimate_device.data(),
      plus_float2_t(),
      context);

    float2 energy_estimate_host = mgpu::from_mem(energy_estimate_device)[0];
    float energy_estimate = energy_estimate_host.y / energy_estimate_host.x;
    printf("%f %d %f\n", energy_estimate, num_walkers, target_energy);

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
    target_energy += damping_alpha * (log(target_num_walkers) - log(num_walkers));
  }  
  return 0;
}

// nvcc -gencode arch=compute_52,code=sm_52 -std=c++11 -I libs/moderngpu/src --expt-extended-lambda -Xptxas="-v" -lineinfo
