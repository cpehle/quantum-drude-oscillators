// -*- c++ -*-
#include <samplers/traditional-dmc.hxx>
#include <samplers/resampling-dmc.hxx>

#include <hamiltonians/qdo-diatom.hxx>

int main(int argc, char** argv) {
  uint thermalization_iters = 5000;

  assert(argc == 5);
  uint niters = atoi(argv[1]);
  int target_num_walkers = atoi(argv[2]);
  float radial_distance = std::atof(argv[3]);
  float charge = std::atof(argv[4]);
  
  mgpu::standard_context_t context(false);

  using system = qdo_atom_dimer;

  using Sampler = TraditionalDMC<system>;

  Sampler dd(target_num_walkers, context);
  dd.initialize();

  for(uint iter = 0; iter < thermalization_iters; ++iter)
    dd.step(system::parameter_t{radial_distance, charge});

  double total_energy = 0.0, total_squared_energy = 0.0;
  for(uint iter = 0; iter < niters; ++iter) {
    float energy = dd.step(system::parameter_t{radial_distance, charge});
    total_energy += energy;
    total_squared_energy += energy*energy;
  }

  double energy_mean = total_energy/niters;
  double energy_variance = total_squared_energy/niters - energy_mean*energy_mean;
  
  printf("%.16f %.16f\n", energy_mean, energy_variance);
  return 0;
}
