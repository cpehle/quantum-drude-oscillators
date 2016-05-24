// -*- c++ -*-
#include <moderngpu/transform.hxx>
#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/kernel_load_balance.hxx>

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
  mgpu::standard_context_t context;
  int seed = 10;
  int target_num_walkers = 100;
  float dt = 0.01;

  // for(int iter = 0; iter < 1 million; ++iter)
  
  int num_walkers = target_num_walkers;
  float target_energy = 0.0;

  mgpu::mem_t<walker_state_t> old_walker_state(num_walkers, context);
  auto old_walker_state_data = old_walker_state.data();
  
  // Initialize all walker positions to 0
  mgpu::transform(
    [=]MGPU_DEVICE(int index) {
      for(int ii = 0; ii < dim; ++ii)
        old_walker_state_data[index].pos[ii] = 0;
    }, num_walkers, context);

  // Diffusion: add a random gaussian to each walker's position
  // TODO: times sqrt_t
  mgpu::transform(
    [=]MGPU_DEVICE(int index) {
      curandStatePhilox4_32_10_t state;
      
      // ask JohnS if I can just seed++ each iteration, maybe sequence++
      curand_init(seed, 0, index, &state); 
      uint4 result = curand4(&state);

      float2 hi = _curand_box_muller(result.x, result.y);
      float2 lo = _curand_box_muller(result.z, result.w);

      old_walker_state_data[index].pos[0] += hi.x;
      old_walker_state_data[index].pos[1] += hi.y;
      old_walker_state_data[index].pos[2] += lo.x;
      // old_walker_state_data[index].pos[3] += lo.y;
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

  mgpu::transform(
    [=]MGPU_DEVICE(int index) {
      children_data[index] = exp(-dt * (energy_data[index] - target_energy));
      // TODO plus a uniformly distributed integer in [0,1]
    }, num_walkers, context);

  // Birth-death II: compute a prefix-sum of the number-of-copies for
  // each walker
  mgpu::mem_t<int> children_offsets(num_walkers, context);
  mgpu::mem_t<int> total_children(1, context);
  mgpu::scan(children.data(), num_walkers, children_offsets.data(),
             mgpu::plus_t<int>(), total_children.data(), context);

  // Birth-death III: Now create children-number of copies of each
  // walker into a second array.
  int num_children = from_mem(total_children)[0];

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
  
  return 0;
}
