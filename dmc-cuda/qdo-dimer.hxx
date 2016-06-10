#include "quantum-system.hxx"

struct qdo_atom_dimer : quantum_system_t<6, 1> {
  using walker_state_t = typename quantum_system_t<6, 1>::walker_state_t;
  using parameter_t = typename quantum_system_t<6, 1>::parameter_t;
  
  MGPU_DEVICE static float local_energy(walker_state_t state, parameter_t parameters) {
    // Distance between two nuclei
    float r12 = parameters[0];

    // Displacement between the two nuclei: vector from drude1 to drude2
    math::vector_t<3> displacement_12{r12, 0, 0};

    // The two drudes in their local frames
    math::vector_t<3> drude1{state[0], state[1], state[2]};
    math::vector_t<3> drude2{state[3], state[4], state[5]};

    // Distance between the two drudes
    float r_d1d2 = sqrt((drude2 + displacement_12 - drude1).norm_squared());

    // Distance between drude2 and nucleus 1
    float r_d2n1 = sqrt((displacement_12 + drude2).norm_squared());

    // Distance between drude1 and nucleus 2
    float r_d1n2 = sqrt((drude1 - displacement_12).norm_squared());

    // Harmonic oscillator energies: x^2/2 for each drude in its local
    // frame
    float energy = drude1.norm_squared()/2 + drude2.norm_squared()/2;

    // Drude-drude interaction energy: erf(r)/r, which is a
    // regularized version of 1/r. Positive because it is repulsive.
    energy += erf(r_d1d2)/r_d1d2;

    // Drude1 - nucleus2 interaction energy: negative because it's attractive
    energy -= erf(r_d1n2)/r_d1n2;

    // Drude2 - nucleus1 interaction energy
    energy -= erf(r_d2n1)/r_d2n1;
    
    return energy;
  }

  MGPU_DEVICE static walker_state_t drift_velocity(walker_state_t state,
                                                   parameter_t parameters) {
    return state * 0.0;
  }
};
