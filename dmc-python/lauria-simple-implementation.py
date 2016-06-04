#!/usr/bin/env python

import numpy as np

class Walker(object):
    def __init__(self, pos, potential):
        self.pos = np.copy(pos)
        self.V = potential
        
    def step(self, dt, target_energy):
        self.pos += np.sqrt(dt) * np.random.randn(*self.pos.shape)
        q = np.exp(- dt * (self.V(self.pos) - target_energy))
        if q - int(q) > np.random.random(): return 1 + int(q)
        return int(q)
        
class DMC(object):
    def __init__(self, target_nwalkers, potential, dimension=(1,3), target_guess=0.0):
        self.walkers = []
        for ii in xrange(target_nwalkers):
            self.walkers.append(Walker(np.random.random(dimension) - 0.5, potential))
        self.target_energy = target_guess
        self.target_nwalkers = np.float64(target_nwalkers)
        self.V = potential
        
    def timestep(self, dt):
        new_walkers = []
        for walker in self.walkers:
            ncopies = walker.step(dt, self.target_energy)
            for jj in xrange(ncopies):
                new_walkers.append(Walker(walker.pos,self.V))
        self.walkers = new_walkers
        self.target_energy += 0.1*(np.log(self.target_nwalkers/len(self.walkers)))
        return self.target_energy

def SHO_3d_single(pos):
    return 0.5 * np.sum(pos**2)

if __name__ == "__main__":
    dd = DMC(300,potential=SHO_3d_single, dimension=(1,3))
    es = []
    for ii in xrange(900):
        es.append(dd.timestep(0.01))
        print es[-1]
