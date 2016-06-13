#include "qdo-diatom.hxx"

/*
  An equilateral triangle of atoms.
*/
struct qdo_atom_trimer : quantum_system_t<9,2> {
  using walker_state_t = typename quantum_system_t<9, 2>::walker_state_t;
  using parameter_t = typename quantum_system_t<9, 2>::parameter_t;

  static constexpr num_t cos30 = 0.8660254037844386467637231707529361834714; // sqrt(3)/2
  
  MGPU_DEVICE static num_t local_energy(walker_state_t state, parameter_t parameters) {
    // Distance between each pair of nuclei
    num_t r = parameters[0];

    // Magnitude of charge on each drude and nucleus
    num_t q = parameters[1];
  
    // The three drudes in their local frames
    math::vector_t<3> drude1{state[0], state[1], state[2]};
    math::vector_t<3> drude2{state[3], state[4], state[5]};
    math::vector_t<3> drude3{state[6], state[7], state[8]};

    // Positions of the three nuclei, defining an equilateral triangle
    // of side r.

    // Counter-clockwise from a nucleus on the negative x-axis, we
    // have nuclei 1,2 and 3.
    math::vector_t<3> nucleus1{-r/2, 0, 0};
    math::vector_t<3> nucleus2{r/2, 0, 0};
    math::vector_t<3> nucleus3{0, r*cos30, 0};

    // Harmonic oscillator monomer energies: known exactly because of
    // our gaussian guide wavefunction
    num_t energy = 4.5;

    energy += qdo_pair_interaction(drude1, drude2, nucleus2-nucleus1, q, q);
    energy += qdo_pair_interaction(drude1, drude3, nucleus3-nucleus1, q, q);
    energy += qdo_pair_interaction(drude2, drude3, nucleus3-nucleus2, q, q);
    
    return energy;
  }

  MGPU_DEVICE static walker_state_t drift_velocity(walker_state_t state,
                                                   parameter_t parameters) {
    return -state;
  }
};
