// -*- c++ -*-
#include <moderngpu/transform.hxx>
#include <curand_kernel.h>

#include "random-numbers.hxx"

template<int Dimension_>
struct quantum_system_t {
  enum { Dimension = Dimension_};

  struct alignas(8) walker_state_t {
    float pos[Dimension];
  };
};

template<int Dim>
struct harmonic_oscillator : quantum_system_t<Dim> {
  using walker_state_t = typename quantum_system_t<Dim>::walker_state_t;
  MGPU_DEVICE static float local_energy(walker_state_t state) {
      return 0;
  }
};

template<typename tt>
struct DMC {
  using system_t = tt;
  using walker_state_t = typename system_t::walker_state_t;
  
  uint seed = 10;
  float dt = 0.005;
  float damping_alpha = 0.1;
  float sqrt_dt = sqrt(dt);
  uint num_walkers;
  uint iter = 0;
  float target_energy = 0.0;
  int target_num_walkers;
  
  DMC(int target) :
    num_walkers(target),
    target_num_walkers(target)
  {}

  void initialize() {
    printf("%d %d\n", system_t::Dimension, num_walkers);
    mgpu::standard_context_t context;
    // Initialize all walker positions to random gaussians of std 1
    mgpu::transform(
      [=]MGPU_DEVICE(uint index) {
          //auto randoms = gpu_random::uniforms<12>(uint4{index, 0, 0, 0}, uint2{seed, 0});
          uint4 result = curand_Philox4x32_10(uint4{index, 0, 0, 0}, uint2{seed, 0});
          //uint4 result{index,seed,3,4};
          printf("HIII %d\n", result.x);
      }, num_walkers, context);

    context.synchronize();
  }

  float step() {
  }
};

int main(int argc, char** argv) {
  assert(argc == 3);
  uint niters = atoi(argv[1]);
  int target_num_walkers = atoi(argv[2]);
  
  DMC<harmonic_oscillator<3>> dd(target_num_walkers);
  dd.initialize();
    
  for(uint iter = 0; iter < niters; ++iter) {
    dd.step();
  }  
  return 0;
}

// nvcc -gencode arch=compute_52,code=sm_52 -std=c++11 -I libs/moderngpu/src --expt-extended-lambda -Xptxas="-v" -lineinfo
