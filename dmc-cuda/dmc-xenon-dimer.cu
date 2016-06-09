// -*- c++ -*-
#include "traditional-dmc.hxx"
#include "qdo-dimer.hxx"

int main(int argc, char** argv) {
  assert(argc == 3);
  uint niters = atoi(argv[1]);
  int target_num_walkers = atoi(argv[2]);
  
  mgpu::standard_context_t context;

  using system = qdo_atom_dimer;
  
  TraditionalDMC<system> dd(target_num_walkers, context);
  dd.initialize();
    
  for(uint iter = 0; iter < niters; ++iter) {
    dd.step(system::parameter_t{10.0});
  }  
  return 0;
}
