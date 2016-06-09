#pragma once

#include <moderngpu/meta.hxx>

namespace math {
  template <int Arity, typename num_t = float>
  struct alignas(8) vector_t {
    num_t values[Arity];

    MGPU_HOST_DEVICE num_t & operator[] (size_t n) {
      return values[n];
    }

    MGPU_HOST_DEVICE vector_t<Arity,num_t> & operator +(vector_t<Arity,num_t> const & other) const {
      vector_t<Arity,num_t> answer;
      mgpu::iterate<Arity>([&](uint ii) {
          answer[ii] = values[ii] + other[ii];
        });
      return answer;
    }
  };
}
    
struct plus_float2_t : public std::binary_function<float2, float2, float2> {
  MGPU_HOST_DEVICE float2 operator()(float2 a, float2 b) const {
    return float2{a.x + b.x, a.y + b.y};
  }
};
