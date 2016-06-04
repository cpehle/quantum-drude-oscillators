#!/usr/bin/env python

import numpy as np

class Walker(object):
    def __init__(self, pos):
        self.pos = np.copy(pos)
        
    def step(self, dt, target_energy):
        E_before = self.local_energy(self.pos)
        self.pos += np.sqrt(dt) * np.random.randn(*self.pos.shape) + dt*self.quantum_force(self.pos)
        E_after = self.local_energy(self.pos)

        branching_factor = np.exp(-dt * (0.5*(E_after + E_before) - target_energy))
        return branching_factor
        
class DMC(object):
    def __init__(self, walker_class, target_nwalkers, dimension=(1,3), target_guess=0.0):
        self.walker_class = walker_class
        self.target_energy = target_guess
        self.target_nwalkers = np.float64(target_nwalkers)
        self.initialize(dimension)

    def initialize(self, dimension):
        self.walkers = []
        for ii in xrange(int(self.target_nwalkers)):
            self.walkers.append(self.walker_class(np.random.random(dimension) - 0.5))
        
    def timestep(self, dt):
        energy = 0.
        total_weight = 0.
        
        new_walkers = []
        for walker in self.walkers:
            branching_factor = walker.step(dt, self.target_energy)

            energy += branching_factor * walker.local_energy(walker.pos)
            total_weight += branching_factor
            
            ncopies = int(branching_factor + np.random.random())
            for jj in xrange(ncopies):
                new_walkers.append(self.walker_class(walker.pos))
        self.walkers = new_walkers
        self.target_energy += 0.1*(np.log(self.target_nwalkers/len(self.walkers)))
        return energy/total_weight

# pick trial wavefunction phi_t(x) = pi^(-1/4) exp(-x^2/2)
class PerfectlySampledHarmonicOscillator(Walker):
    @staticmethod
    def local_energy(pos):
        # (H phi_t)/phi_t
        # where H = (-1/2)d^2/dx^2 + (1/2)x^2
        # Since we chose the exactly correct answer, it turns out to
        # be a constant.
        return 0.5 * pos.shape[0]

    @staticmethod
    def quantum_force(pos):
        # (grad phi_t)/phi_t
        return -pos

# phi_t(x) = 1
class HarmonicOscillator(Walker):
    @staticmethod
    def local_energy(pos):
        # (H 1)/1
        # = (-1/2)d^2/dx^2(1) + 1/2x^2
        # = x^2/2
        return pos.dot(pos)/2

    @staticmethod
    def quantum_force(pos):
        # (grad 1)/1
        return np.zeros_like(pos)

# phi_t(x) = sqrt(alpha) pi^(-1/4) exp(-alpha^2 x^2 / 2)
# This interpolates between the above two samplers: at alpha=0, it
# reproduces the naive approach with no importance sampling. At
# alpha=1, it reproduces perfect sampling with no variance. At
# intermediate values, we get intermediate variances.
def ImportanceSampledHarmonicOscillator(alpha):
    class Oscillator(Walker):
        @staticmethod
        def local_energy(pos):
            term = alpha**2 + (1 - alpha**4) * pos.dot(pos)
            return term/2

        @staticmethod
        def quantum_force(pos):
            return -pos * alpha**2

    return Oscillator        

def main(args):
    dd = DMC(ImportanceSampledHarmonicOscillator(args.alpha),
             args.nwalkers,
             dimension=(1,))

    for ii in xrange(args.nsteps):
        print dd.timestep(args.timestep), len(dd.walkers)

def opts():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--timestep", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Controls the quality of trial wavefunction. 1 is perfect guidance, 0 is no guidance")
    parser.add_argument("--nwalkers", type=int, default=300)
    parser.add_argument("--nsteps", type=int, default=1000)
    return parser

if __name__ == "__main__":
    main(opts().parse_args())
