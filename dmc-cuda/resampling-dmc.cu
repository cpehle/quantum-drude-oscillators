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

  /* Compute the weight for every walker. */
  auto weights_data = weights.data();
  mgpu::transform_reduce(
    [=]MGPU_DEVICE(uint index) {
      float weight = index*index;
      weights_data[index] = weight;
      return weight;
    },
    num_walkers,
    total_weight.data(),
    mgpu::plus_t<float>(),
    context);

  float total_weight_host = mgpu::from_mem(total_weight)[0];

  /* Compute the prefix-sum for weights. */
  mgpu::mem_t<float> should_be_one(1, context);
  mgpu::mem_t<float> weights_prefix_sum(num_walkers, context);
  mgpu::transform_scan<float,mgpu::scan_type_t::scan_type_inc>(
    [=]MGPU_DEVICE(uint index) {
      return weights_data[index]/total_weight_host;
    },
    num_walkers,
    weights_prefix_sum.data(),
    mgpu::plus_t<float>(),
    should_be_one.data(),
    context);
  
  float one = mgpu::from_mem(should_be_one)[0];
  printf("%f\n", one);

  /*
    as in, if you want to have 4 walkers, then use the points (0.2+x,0.4+x,0.6+x,0.8+x), where x is uniform between 0 and 0.2
  */
  mgpu::mem_t<float> probabilities(num_walkers, context);
  auto probabilities_data = probabilities.data();
  mgpu::transform(
    [=]MGPU_DEVICE(uint index) {
      //float random = gpu_random::uniforms<1>(uint4{index, 0, 0, 0}, uint2{seed, 1})[0];
      float random = float(index)/num_walkers;
      probabilities_data[index] = random;
    }, num_walkers, context);

  mgpu::mergesort(probabilities.data(), num_walkers, mgpu::less_t<float>(), context);

  mgpu::mem_t<int> indices(num_walkers, context);
  mgpu::sorted_search<mgpu::bounds_lower>(
    probabilities.data(), num_walkers,
    weights_prefix_sum.data(), num_walkers,
    mgpu::make_store_iterator<int>([=]MGPU_DEVICE(uint parent_index, uint child_index) {
        printf("%d %d\n", child_index, parent_index);
      }),
    mgpu::less_t<float>(),
    context);
  
  std::vector<float> ww = mgpu::from_mem(weights);
  std::vector<float> prefixes = mgpu::from_mem(weights_prefix_sum), probs = mgpu::from_mem(probabilities);
  //std::vector<int> inds    = mgpu::from_mem(indices);

  for(int ii = 0; ii < num_walkers; ++ii) {
    printf("%f %f %f\n", ww[ii], prefixes[ii], probs[ii]);
  }
 
  return 0;
}
