// -*- c++ -*-
#include <moderngpu/kernel_mergesort.hxx>
#include <moderngpu/kernel_sortedsearch.hxx>
#include <moderngpu/transform.hxx>
#include <moderngpu/kernel_reduce.hxx>
#include <moderngpu/kernel_scan.hxx>

#include "random-numbers.hxx"

int main(int argc, char **argv) {
  mgpu::standard_context_t context;
  uint seed = 11;
  uint num_walkers = 20;

  mgpu::mem_t<float> weights(num_walkers, context);
  mgpu::mem_t<float> total_weight(1, context);
  
  auto weights_data = weights.data();
  mgpu::transform_reduce(
    [=]MGPU_DEVICE(uint index) {
      //float random = gpu_random::uniforms<1>(uint4{index, 0, 0, 0}, uint2{seed, 0})[0];
      float random = index*index;
      //random *= random;
      weights_data[index] = random;
      return random;
    },
    num_walkers,
    total_weight.data(),
    mgpu::plus_t<float>(),
    context);

  float total_weight_host = mgpu::from_mem(total_weight)[0];
  printf("%f\n", total_weight_host);

  mgpu::transform(
    [=]MGPU_DEVICE(uint index) {
      weights_data[index] = weights_data[index]/total_weight_host;
    },
    num_walkers,
    context);
  
  mgpu::mem_t<float> should_be_one(1, context);
  mgpu::mem_t<float> weights_prefix_sum(num_walkers, context);
  mgpu::scan(weights.data(), num_walkers, weights_prefix_sum.data(),
             mgpu::plus_t<float>(), should_be_one.data(), context);

  float one = mgpu::from_mem(should_be_one)[0];
  printf("%f\n", one);

  mgpu::mem_t<float> probabilities(num_walkers, context);
  auto probabilities_data = probabilities.data();
  mgpu::transform(
    [=]MGPU_DEVICE(uint index) {
      float random = gpu_random::uniforms<1>(uint4{index, 0, 0, 0}, uint2{seed, 1})[0];
      probabilities_data[index] = random;
    }, num_walkers, context);

  mgpu::mergesort(probabilities.data(), num_walkers, mgpu::less_t<float>(), context);

  mgpu::mem_t<int> indices(num_walkers, context);
  mgpu::sorted_search<mgpu::bounds_upper>(probabilities.data(),
                                          num_walkers,
                                          weights_prefix_sum.data(),
                                          num_walkers,
                                          indices.data(),
                                          mgpu::less_t<float>(),
                                          context);

  std::vector<float> ww = mgpu::from_mem(weights);
  std::vector<float> prefixes = mgpu::from_mem(weights_prefix_sum), probs = mgpu::from_mem(probabilities);
  std::vector<int> inds    = mgpu::from_mem(indices);

  for(int ii = 0; ii < num_walkers; ++ii) {
    printf("%f %f %f %d\n", ww[ii], prefixes[ii], probs[ii], inds[ii]);
  }
  
  
  /*
  
  
  auto numbers_data = numbers.data();
  mgpu::transform(
    [=]MGPU_DEVICE(uint index) {
      numbers_data[index] = -float(index % 5);
    }, 20, context);

  std::vector<float> before = mgpu::from_mem(numbers);
  for(auto xx : before)
    printf("%f ", xx);
  printf("\n");

  mgpu::mergesort(numbers.data(), 20, mgpu::less_t<float>(), context);

  std::vector<float> after = mgpu::from_mem(numbers);
  for(auto xx : after)
    printf("%f ", xx);
  printf("\n");
  */  
  return 0;
}
