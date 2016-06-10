// -*- c++ -*-
#include "traditional-dmc.hxx"
#include "qdo-dimer.hxx"

int main(int argc, char** argv) {
  assert(argc == 4);
  uint niters = atoi(argv[1]);
  int target_num_walkers = atoi(argv[2]);
  float radial_distance = std::atof(argv[3]);
  
  mgpu::standard_context_t context;

  using system = qdo_atom_dimer;
  
  TraditionalDMC<system> dd(target_num_walkers, context);
  dd.initialize();

  float total_energy = 0.0, total_squared_energy = 0.0;
  for(uint iter = 0; iter < niters; ++iter) {
    float energy = dd.step(system::parameter_t{radial_distance});
    total_energy += energy;
    total_squared_energy += energy*energy;
  }

  float energy_mean = total_energy/niters;
  float energy_variance = total_squared_energy/niters - energy_mean;
  
  printf("%f %f\n", energy_mean, energy_variance);  
  return 0;
}
