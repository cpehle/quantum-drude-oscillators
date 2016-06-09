#include "quantum-system.hxx"

template<int Dim>
struct harmonic_oscillator : quantum_system_t<Dim,0> {
  using walker_state_t = typename quantum_system_t<Dim,0>::walker_state_t;
  MGPU_DEVICE static float local_energy(walker_state_t state) {
    float xx = 0;
    for(int ii = 0; ii < Dim; ++ii)
      xx += state[ii]*state[ii];
    return xx/2;
  }
};

template<int Dim>
struct importance_sampled_harmonic_oscillator : quantum_system_t<Dim,1> {
  using walker_state_t = typename quantum_system_t<Dim,1>::walker_state_t;
  using parameter_t = typename quantum_system_t<Dim,1>::parameter_t;
  MGPU_DEVICE static float local_energy(walker_state_t state, parameter_t parameters) {
    float alpha = parameters[0];
    float alpha2 = alpha*alpha;
    float alpha4 = alpha2*alpha2;

    float xx = state.norm_squared();

    return (alpha2 + (1 - alpha4)*xx)/2;
  }

  MGPU_DEVICE static walker_state_t drift_velocity(walker_state_t state,
                                                   parameter_t parameters) {
    float alpha = parameters[0];
    float alpha2 = alpha*alpha;
    return -alpha2 * state;
  }
};
