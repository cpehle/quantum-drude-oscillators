#include "vector.hxx"

template<int walker_dimension_, int parameter_dimension_>
struct quantum_system_t {
  enum { walker_dimension = walker_dimension_, parameter_dimension = parameter_dimension_};
  using walker_state_t = math::vector_t<walker_dimension>;
  using parameter_t = math::vector_t<parameter_dimension>;
};
