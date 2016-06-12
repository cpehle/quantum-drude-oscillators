// -*- c++ -*-
#include "traditional-dmc.hxx"
#include "harmonic-oscillator.hxx"

int main(int argc, char** argv) {
  assert(argc == 3);
  uint niters = atoi(argv[1]);
  int target_num_walkers = atoi(argv[2]);
  mgpu::standard_context_t context;

  using system = importance_sampled_harmonic_oscillator<3>;
  
  TraditionalDMC<system> dd(target_num_walkers, context);
  dd.initialize();
    
  for(uint iter = 0; iter < niters; ++iter) {
    dd.step(system::parameter_t{0});
  }  
  return 0;
}

// nvcc -gencode arch=compute_52,code=sm_52 -std=c++11 -I libs/moderngpu/src --expt-extended-lambda -Xptxas="-v" -lineinfo
