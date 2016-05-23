#include <moderngpu/transform.hxx>
#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/kernel_load_balance.hxx>

using namespace mgpu;

struct alignas(8) walker_state_t {
  float yo[6];
};

int main(int argc, char** argv) {
  standard_context_t context;

  int num_walkers = 100;
  mem_t<walker_state_t> old_walker_state(num_walkers, context);

  mem_t<int> children(num_walkers, context);
  int* children_data = children.data();

  transform(
    [=]MGPU_DEVICE(int index) {
      // Some function f(index).
      children_data[index] = 3 & index;
    }, num_walkers, context
  );

  // Do second pass.
  mem_t<int> children_offsets(num_walkers, context);
  mem_t<int> total_children(1, context);
  scan(children.data(), num_walkers, children_offsets.data(),
    plus_t<int>(), total_children.data(), context);

  // Now create children-number of copies of each walker into a second 
  // array.
  int num_children = from_mem(total_children)[0];

  mem_t<int> next_gen(num_children, context);
  int* next_gen_data = next_gen.data();

  // Fill next_gen with the value of the parent walker.
  mem_t<walker_state_t> new_walker_state(num_children, context);
  auto new_state = new_walker_state.data();

  transform_lbs(
    [=]MGPU_DEVICE(int index, int parent, int sibling,
      tuple<walker_state_t> desc) {
      new_state[index] = get<0>(desc);
    }, num_children, children_offsets.data(), num_walkers, 
    make_tuple(old_walker_state.data()),
    context
  );

  old_walker_state.swap(new_walker_state);
  num_walkers = num_children;

  return 0;
}
