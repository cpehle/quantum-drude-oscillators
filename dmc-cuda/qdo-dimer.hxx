#include "quantum-system.hxx"

struct qdo_atom_dimer : quantum_system_t<6, 1> {
  using walker_state_t = typename quantum_system_t<6, 1>::walker_state_t;
  using parameter_t = typename quantum_system_t<6, 1>::parameter_t;
  
  MGPU_DEVICE static float local_energy(walker_state_t state, parameter_t parameters) {
    float r1=0, r2=0, r12=0, tmp;
    for(int ii = 0; ii < 3; ++ii) {
      r1  += state[ii] * state[ii];
      r2  += state[ii+3] * state[ii+3];

      tmp = state[ii+3] - state[ii] + r_centres;
      r12 += tmp*tmp;
    }

    r12 = sqrt(r12);

    float energy = r1/2 + r2/2 + 1/r12;
    energy += 1/(distance between drude1 and centre2);
    energy += 1/(distance between drude2 and centre1);

    return energy;
  }

  MGPU_DEVICE static float drift_velocity(walker_state_t state,
                                          parameter_t parameters) {
    return zero;
  }
