// -*- c++ -*-
#include <moderngpu/kernel_scan.hxx>
#include <iostream>

using num_t = float;

int main(int argc, char **argv) {
  mgpu::standard_context_t context;
  int num_items = 10;
  
  mgpu::mem_t<num_t> offsets(num_items, context);
  mgpu::mem_t<num_t> total(1, context);

  mgpu::transform_scan(
    [=]MGPU_DEVICE(uint index) {
      return num_t(1);
    },
    num_items,
    offsets.data(),
    mgpu::plus_t<num_t>(),
    total.data(),
    context);

  num_t total_host = mgpu::from_mem(total)[0];
  std::cout << total_host << "\n";

  return 0;
}
  
