#!/usr/bin/env python

import numpy as np

def run_DMC(nsteps, nwalkers, Vfunc, dt, DIM=3, target_guess=0.0):
    sqrt_dt = np.sqrt(dt)
    walkers = np.random.random((nwalkers,DIM)) - 0.5
    target_energy = target_guess
    
    energies = []
    
    for ii in xrange(nsteps):
        walkers += sqrt_dt * np.random.randn(*walkers.shape)
        dv = Vfunc(walkers) - target_energy
        ncopies = np.array(np.exp(-dt*dv) + np.random.random(walkers.shape[0]), dtype=np.int)
        nkeep = walkers[np.flatnonzero(ncopies>0),:]
        ndup  = walkers[np.flatnonzero(ncopies>1),:]
        walkers = np.vstack([nkeep,ndup])        
        target_energy += 0.1*(np.log(nwalkers) - np.log(walkers.shape[0]))
        energies.append(target_energy)
        
    return energies
