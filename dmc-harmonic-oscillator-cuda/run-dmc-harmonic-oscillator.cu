// -*- c++ -*-
#include <moderngpu/transform.hxx>
#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/kernel_load_balance.hxx>
#include <moderngpu/kernel_reduce.hxx>

#include <curand_kernel.h>
#include <curand_normal.h>

const int dim = 3;

struct alignas(8) walker_state_t {
  float pos[dim];
};

MGPU_DEVICE float harmonic_oscillator_hamiltonian(walker_state_t state) {
  float xx;
  for(int ii = 0; ii < dim; ++ii)
    xx += state.pos[ii]*state.pos[ii];
  return xx/2;
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
  
  // Initialize all walker positions to 0
  mgpu::transform(
    [=]MGPU_DEVICE(int index) {
      for(int ii = 0; ii < dim; ++ii)
        old_walker_state_data[index].pos[ii] = 0;
    }, num_walkers, context);
    
  for(uint iter = 0; iter < niters; ++iter) {
    printf("%d %f\n", num_walkers, target_energy);
    old_walker_state_data = old_walker_state.data();

    // Diffusion: add a random gaussian of stddev dt to each walker's position
    mgpu::transform(
      [=]MGPU_DEVICE(uint index) {
        uint4 result = curand_Philox4x32_10(uint4{index, iter, 0, 0}, uint2{seed, 0});
        
        float2 hi = _curand_box_muller(result.x, result.y);
        float2 lo = _curand_box_muller(result.z, result.w);

        old_walker_state_data[index].pos[0] += sqrt_dt * hi.x;
        old_walker_state_data[index].pos[1] += sqrt_dt * hi.y;
        old_walker_state_data[index].pos[2] += sqrt_dt * lo.x;

      }, num_walkers, context);

    // Energy evaluation: evaluate the Hamiltonian at each walker's
    // position
    mgpu::mem_t<float> energy(num_walkers, context);
    auto energy_data = energy.data();
  
    mgpu::transform(
      [=]MGPU_DEVICE(int index) {
        energy_data[index] = harmonic_oscillator_hamiltonian(old_walker_state_data[index]);
      }, num_walkers, context);
    
    // Birth-death I: calculate the number of copies of each walker in the
    // next generation
    mgpu::mem_t<int> children(num_walkers, context);
    int* children_data = children.data();

    mgpu::mem_t<float> energy_estimate(1, context);
    mgpu::transform_reduce(
      [=]MGPU_DEVICE(uint index) {
        float branching_factor = exp(-dt * (energy_data[index] - target_energy));
        uint4 rand_result = curand_Philox4x32_10(uint4{index, iter, 1, 0}, uint2{seed, 0});
        float uniform_float = _curand_uniform(rand_result.x);

        children_data[index] = int(branching_factor + uniform_float);

        return branching_factor * energy_data[index];
      }, num_walkers, energy_estimate.data(), mgpu::plus_t<float>(), context);

    // Birth-death II: compute a prefix-sum of the number-of-copies for
    // each walker
    mgpu::mem_t<int> children_offsets(num_walkers, context);
    mgpu::mem_t<int> total_children(1, context);
    mgpu::scan(children.data(), num_walkers, children_offsets.data(),
               mgpu::plus_t<int>(), total_children.data(), context);

    // Birth-death III: Now create children-number of copies of each
    // walker into a second array.
    int num_children = from_mem(total_children)[0];
    assert(num_children > 0);

    mgpu::mem_t<int> next_gen(num_children, context);
    int* next_gen_data = next_gen.data();

    // Fill next_gen with the value of the parent walker.
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
