#include "quantum-system.hxx"

struct helium : quantum_system_t<6,1> {
  MGPU_DEVICE static float local_energy(walker_state_t state,
                                        parameter_t params) {
    /* r1,r2,r12 = norm(pos[0]), norm(pos[1]), norm(pos[0]-pos[1])
       return 1/r12 - 2/r1 - 2/r2
    */

    float r1=0, r2=0, r12=0, tmp;
    for(int ii = 0; ii < 3; ++ii) {
      r1  += state[ii] * state[ii];
      r2  += state[ii+3] * state[ii+3];

      tmp = state[ii+3] - state[ii];
      r12 += tmp*tmp;
    }
    r1 = sqrt(r1); r2 = sqrt(r2); r12 = sqrt(r12);
    return 1/r12 - 2/r1 - 2/r2;
  }
};
