#pragma once

#include "quantum-system.hxx"

using num_t = float;

/*
  Interaction between two QDO atoms, each electrically neutral.

  drude1: vector from nucleus1 to drude1.
  drude2: vector from nucleus2 to drude2.
  displacement12: vector from nucleus1 to nucleus2

  q1: charge on nucleus1, -charge on drude1
  q2: charge on nucleus2, -charge on drude2
*/
MGPU_HOST_DEVICE
num_t qdo_pair_interaction(math::vector_t<3> drude1, math::vector_t<3> drude2,
                           math::vector_t<3> displacement_12,
                           num_t q1, num_t q2) {
  // Distance between the two drudes
  num_t r_d1d2 = sqrt((drude2 + displacement_12 - drude1).norm_squared());

  // Distance between drude2 and nucleus 1
  num_t r_d2n1 = sqrt((displacement_12 + drude2).norm_squared());

  // Distance between drude1 and nucleus 2
  num_t r_d1n2 = sqrt((drude1 - displacement_12).norm_squared());

  num_t energy = 0.0;
  
  // Drude-drude interaction energy. Positive because it is repulsive.
  energy += q1*q2/r_d1d2;

  // Nucleus-nucleus interaction energy. A constant but
  // r12-dependent contribution.
  energy += q1*q2/sqrt(displacement_12.norm_squared());

  // Drude1 - nucleus2 interaction energy: negative because it's attractive
  energy -= q1*q2/r_d1n2;

  // Drude2 - nucleus1 interaction energy
  energy -= q1*q2/r_d2n1;
    
  return energy;
}

struct qdo_atom_dimer : quantum_system_t<6, 2> {
  using walker_state_t = typename quantum_system_t<6, 2>::walker_state_t;
  using parameter_t = typename quantum_system_t<6, 2>::parameter_t;
  
  MGPU_DEVICE static float local_energy(walker_state_t state, parameter_t parameters) {
    // Distance between two nuclei
    float r12 = parameters[0];

    // Magnitude of charge on each drude and nucleus
    float q = parameters[1];

    // Displacement between the two nuclei: vector from nucleus1 to nucleus2
    math::vector_t<3> displacement_12{r12, 0, 0};

    // The two drudes in their local frames
    math::vector_t<3> drude1{state[0], state[1], state[2]};
    math::vector_t<3> drude2{state[3], state[4], state[5]};

    // Harmonic oscillator energies: x^2/2 for each drude in its local
    // frame
    // float energy = drude1.norm_squared()/2 + drude2.norm_squared()/2;
    // Since we're using importance sampling, this is known exactly.
    float energy = 3.0;

    energy += qdo_pair_interaction(drude1, drude2, displacement_12, q, q);
    
    return energy;
  }

  MGPU_DEVICE static walker_state_t drift_velocity(walker_state_t state,
                                                   parameter_t parameters) {
    return -state;
  }
};
