#include <curand_kernel.h>
#include <curand_normal.h>

#include <moderngpu/transform.hxx>
#include <moderngpu/kernel_reduce.hxx>
#include <moderngpu/memory.hxx>

#include <cstdlib>
#include <cassert>

int main(int argc, char** argv) {
  assert(argc == 2);
  mgpu::standard_context_t context;

  int count = 1000;
  int seed = atoi(argv[1]);

  mgpu::mem_t<float2> outputs(count, context);
  auto outputs_data = outputs.data();

  mgpu::transform(
    [=]MGPU_DEVICE(int index) {
      curandStatePhilox4_32_10_t state;

      curand_init(seed, 0, index, &state);
      uint4 result = curand4(&state);

      float2 yo = _curand_box_muller(result.x, result.y);
      outputs_data[index] = yo;
    },
    count,
    context
  );

  std::vector<float2> host = from_mem(outputs);
  for(float2 p : host) {
    printf("% 13.3e % 13.3e\n", p.x, p.y);
  }

  return 0;
}

// nvcc -arch sm_52 -std=c++11 -I libs/moderngpu/src --expt-extended-lambda -Xptxas="-v" -o curand_test curand_test.cu
