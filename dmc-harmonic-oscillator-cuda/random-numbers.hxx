#include <curand_kernel.h>
#include <curand_normal.h>

namespace gpu_random {
  template<int Arity>
  struct array_float_t {
    float values[Arity];
    MGPU_DEVICE float operator[] (size_t n) const {
      return values[n];
    }
  };

  template<int Arity>
  MGPU_DEVICE array_float_t<Arity> gaussians(uint4 counter, uint2 key) {
    enum { n_blocks = (Arity + 4 - 1)/4 };

    float scratch[n_blocks * 4];
  
    mgpu::iterate<n_blocks>([&](uint index) {
        uint4 local_counter = counter; local_counter.w = index;
        uint4 result = curand_Philox4x32_10(local_counter, key);

        float2 hi = _curand_box_muller(result.x, result.y);
        float2 lo = _curand_box_muller(result.z, result.w);

        uint ii = index*4;
        scratch[ii] = hi.x;
        scratch[ii+1] = hi.y;
        scratch[ii+2] = lo.x;
        scratch[ii+3] = lo.y;
      });

    array_float_t<Arity> answer;

    mgpu::iterate<Arity>([&](uint index) {
        answer.values[index] = scratch[index];
      });
  
    return answer;
  }

  template<int Arity>
  MGPU_DEVICE array_float_t<Arity> uniforms(uint4 counter, uint2 key) {
    enum { n_blocks = (Arity + 4 - 1)/4 };

    float scratch[n_blocks * 4];
  
    mgpu::iterate<n_blocks>([&](uint index) {
        uint4 local_counter = counter; local_counter.w = index;
        uint4 result = curand_Philox4x32_10(local_counter, key);

        uint ii = index*4;
        scratch[ii]   = _curand_uniform(result.x);
        scratch[ii+1] = _curand_uniform(result.y);
        scratch[ii+2] = _curand_uniform(result.z);
        scratch[ii+3] = _curand_uniform(result.w);
      });

    array_float_t<Arity> answer;

    mgpu::iterate<Arity>([&](uint index) {
        answer.values[index] = scratch[index];
      });
  
    return answer;
  }  
}
