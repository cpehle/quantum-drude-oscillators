#include <cstdio>

#include <cmath>
#include <random>

#include <cassert>

#include "lcg.h"

template <class T>
void boxmuller(T* data, size_t count) {
  assert(count % 2 == 0);
  static const T twopi = T(2.0 * 3.14159265358979323846);

  static LCG<T> r;
  for (size_t i = 0; i < count; i += 2) {
    T u1 = 1.0f - r(); // [0, 1) -> (0, 1]
    T u2 = r();
    T radius = std::sqrt(-2 * std::log(u1));
    T theta = twopi * u2;
    data[i    ] = radius * std::cos(theta);
    data[i + 1] = radius * std::sin(theta);
  }
}

using real_t = float;

int main(int argc, char **argv) {
  long n_interactions;
  assert(argc == 2);
  sscanf(argv[1], "%ld", &n_interactions);
  printf("%ld\n", n_interactions);

  real_t x1 = 1.0, y1 = 2.0, z1 = 3.0;
  real_t x2 = 2.0, y2 = 3.0, z2 = 4.0;

  real_t randoms[6];

  real_t answer = 0;
  
  for(long ii = 0; ii < n_interactions; ++ii) {
    real_t dx = x1-x2, dy = y1-y2, dz = z1-z2;
    real_t r = std::sqrt(dx*dx + dy*dy + dz*dz);
    real_t en = std::erf(r);

    boxmuller(&randoms[0], 6);
    x1 += randoms[0];
    x2 += randoms[1];    
    y1 += randoms[2];
    y2 += randoms[3];    
    z1 += randoms[4];
    z2 += randoms[5];    

    answer += en;
  }

  printf("%f\n", answer);
  return 0;
}

/*
  500,000 steps * 1000 walkers * 8 interactions per dimer = 4,000,000,000 = 4billion interactions
  
time ./a.out 100,000,000
100000000
16777216.000000

real    0m13.045s
user    0m13.037s
sys     0m0.001s

So 130 seconds for 1billion interactions ie. 520 seconds for 3billion interactions ie. 8.66 minutes

This is an overestimate: we are generating more random numbers than we
need, and the cost of generating them is the biggest cost here. In
reality, we generate 6 random numbers (2 drudes x 3 dimensions) and
then compute 8 interactions (drude1xdrude2,o2,h2_a, h2_b).
 */
