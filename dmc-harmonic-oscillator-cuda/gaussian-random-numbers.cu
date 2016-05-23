#include <curand_kernel.h>
#include <curand_normal.h>
#include <moderngpu/transform.hxx>
#include <moderngpu/kernel_reduce.hxx>
#include <moderngpu/memory.hxx>
#include <cstdlib>

int main(int argc, char** argv) {

  mgpu::standard_context_t context;

  int count = 100000000;
  int seed = atoi(argv[1]);

  mgpu::mem_t<float2> outputs(1, context);

  mgpu::transform_reduce(
    [=]MGPU_DEVICE(int index) {
      curandStatePhilox4_32_10_t state;

      curand_init(seed, 0, index, &state);
      uint4 result = curand4(&state);

      float2 yo = _curand_box_muller(result.x, result.y);
      return yo;

    }, count, outputs.data(), 
      []MGPU_DEVICE(float2 a, float2 b) {
        return make_float2(a.x + b.x, a.y + b.y);
      }, context
  );

  std::vector<float2> host = from_mem(outputs);
  for(float2 p : host) {
    printf("% 13.3e % 13.3e\n", p.x / count, p.y / count);
  }

  return 0;
}

// nvcc -arch sm_52 -std=c++11 -I libs/moderngpu/src --expt-extended-lambda -Xptxas="-v" -o curand_test curand_test.cu
