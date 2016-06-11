// -*- c++ -*-
#include <moderngpu/kernel_mergesort.hxx>
#include <moderngpu/kernel_sortedsearch.hxx>
#include <moderngpu/transform.hxx>
#include <moderngpu/kernel_reduce.hxx>
#include <moderngpu/kernel_scan.hxx>

#include "random-numbers.hxx"

//#include "harmonic-oscillator.hxx"
//#include "helium.hxx"
#include "qdo-dimer.hxx"

using num_t = double;

//using system_t = importance_sampled_harmonic_oscillator<3>;
//using system_t = helium;
using system_t = qdo_atom_dimer;
using walker_state_t = typename system_t::walker_state_t;
using parameter_t = typename system_t::parameter_t;

int main(int argc, char **argv) {
  mgpu::standard_context_t context;
  uint seed = 11;

  assert(argc == 3);
  uint niters = atoi(argv[1]);
  int num_walkers = atoi(argv[2]);

  double dt = 0.01;
  double sqrt_dt = sqrt(dt);

  auto guide_parameters = parameter_t{2.0};

  mgpu::mem_t<walker_state_t> old_walker_state(num_walkers, context);
  mgpu::mem_t<walker_state_t> new_walker_state(num_walkers, context);

  // Initialize walker positions to random gaussians of std 1
  auto old_walker_state_data = old_walker_state.data();
  mgpu::transform(
    [=]MGPU_DEVICE(uint index) {
      auto randoms =
        gpu_random::gaussians<system_t::walker_dimension>(uint4{index, 0, 0, 0},
                                                         uint2{seed, 0});
      old_walker_state_data[index] = randoms;
    }, num_walkers, context);  
  
  mgpu::mem_t<double> weights(num_walkers, context);
  mgpu::mem_t<double> probabilities(num_walkers, context);
  mgpu::mem_t<double> weights_prefix_sum(num_walkers, context);

  double total_energy = 0.0, total_squared_energy = 0.0;
  
  for (uint local_iter = 0; local_iter < niters; ++local_iter) {
    old_walker_state_data = old_walker_state.data();
    mgpu::mem_t<double2> energy_estimate_device(1, context);
  
    /* Perform diffusion and compute the weight for every walker. */
    auto weights_data = weights.data();
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
        mgpu::iterate<system_t::walker_dimension>([&](uint dimension_index) {
            walker_state[dimension_index] += sqrt_dt * diffusion_randoms.values[dimension_index]
              + dt * drift[dimension_index];
          });
        auto energy_after = system_t::local_energy(walker_state, guide_parameters);

        old_walker_state_data[index] = walker_state;
        
        double weight = exp(-dt * (0.5*(energy_before + energy_after)));
        weights_data[index] = weight;
        return double2{weight, weight * energy_after};
      },
      num_walkers,
      energy_estimate_device.data(),
      plus_double2_t(),
      context);
    double2 energy_estimate_host = mgpu::from_mem(energy_estimate_device)[0];
    assert(!std::isnan(energy_estimate_host.x));
    assert(std::isfinite(energy_estimate_host.x));
    assert(!std::isnan(energy_estimate_host.y));
    assert(std::isfinite(energy_estimate_host.y));
    assert(energy_estimate_host.x != 0);
    
    double energy_estimate = energy_estimate_host.y / energy_estimate_host.x;
    //printf("%f %d\n", energy_estimate, num_walkers);    
    assert(!std::isnan(energy_estimate));
    assert(std::isfinite(energy_estimate));

    double total_weight_host = energy_estimate_host.x;

    /* Compute the prefix-sum for weights. */
    mgpu::mem_t<double> should_be_one(1, context);
    mgpu::transform_scan<double,mgpu::scan_type_t::scan_type_inc>(
      [=]MGPU_DEVICE(uint index) {
        return weights_data[index]/total_weight_host;
      },
      num_walkers,
      weights_prefix_sum.data(),
      mgpu::plus_t<double>(),
      should_be_one.data(),
      context);

    //double weight_sum = mgpu::from_mem(weights_prefix_sum)[num_walkers-1];
    //printf("%.20f\n", weight_sum);
  
    double one = mgpu::from_mem(should_be_one)[0];
    assert(abs(one-1.0) < 1e-6);

    auto probabilities_data = probabilities.data();
    mgpu::transform(
      [=]MGPU_DEVICE(uint index) {
        double jitter = gpu_random::uniforms<1>(uint4{index, local_iter, 0, 0}, uint2{seed, 2})[0];
        double random = double(index)/(num_walkers+1) + jitter/num_walkers;
        
        //double random = gpu_random::uniforms<1>(uint4{index, local_iter, 0, 0}, uint2{seed, 2})[0];
        probabilities_data[index] = random;
      }, num_walkers, context);

    //mgpu::mergesort(probabilities.data(), num_walkers, mgpu::less_t<double>(), context);

    auto new_walker_state_data = new_walker_state.data();
    auto weights_prefix_sum_data = weights_prefix_sum.data();
    mgpu::sorted_search<mgpu::bounds_lower>(
      probabilities.data(), num_walkers,
      weights_prefix_sum.data(), num_walkers,
      mgpu::make_store_iterator<int>([=]MGPU_DEVICE(uint parent_index, uint child_index) {
          if(parent_index >= num_walkers) {
            parent_index = num_walkers-1;/*
            printf("%d %d %d %.20f %.20f\n", parent_index, child_index,
                   num_walkers,
                   probabilities_data[child_index],
                   weights_prefix_sum_data[parent_index-1]);*/
          }
          assert(parent_index < num_walkers);

          new_walker_state_data[child_index] = old_walker_state_data[parent_index];
        }),
      mgpu::less_t<double>(),
      context);

    old_walker_state.swap(new_walker_state);

    if(local_iter > 1300) {
      total_energy += energy_estimate;
      total_squared_energy += energy_estimate*energy_estimate;
    }
  }

  niters -= 1300;
  double energy_mean = total_energy/niters;
  double energy_variance = total_squared_energy/niters - energy_mean*energy_mean;
  
  printf("%f %f\n", energy_mean, energy_variance);  
  

  /*
    std::vector<double> ww = mgpu::from_mem(weights);
    std::vector<double> prefixes = mgpu::from_mem(weights_prefix_sum), probs = mgpu::from_mem(probabilities);

    for(int ii = 0; ii < num_walkers; ++ii) {
    printf("%f %f %f\n", ww[ii], prefixes[ii], probs[ii]);
    }
  */
  return 0;
}
