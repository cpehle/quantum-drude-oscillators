Here is a detailed intern project plan. The ultimate goal of this
project is to predict water-methane dimer energies, having fit only to
monomer properties of methane taken by itself. The water model will be
taken from Jones et al. 

No dynamics will be addressed. We are only going to look at energies
of fixed nuclear configurations. We will use Diffusion Monte Carlo
(DMC) to compute these energies.

I have divided (FIXME) the overall project into a series of short
assignments. Each assignment achieves a milestone along the way to a
working implementation of QDO model evaluation. I have marked with an
asterisk (*) what I have done so far, so that the intern(s) can read
my code as they work through that problem.

The first couple of assignments aim to familiarize the intern(s) with
Python, classical forcefield models, working with quantum data. ,
and the problem of building forcefield models from ab initio data.

1. *Compute and plot xenon dimer radial scan energies using point
charges and LJ12-6.

2. *Compute and plot dimer energies for radial and rotational scans of
TIP3P and TIP4P water, in Python.

3. *Plot QM water dimer scan data: total, electrostatic, dispersion and
exchange.

4. Fit xenon vdw parameters from QM data.

The next couple of assignments aim to introduce the intern(s) to
DMC, quantum mechanics.

5. *Write DMC in Python for the harmonic oscillator in 1,2 and 3
dimensions.

6. *Rewrite the DMC implementation using numpy vectorized operations.

7. *Write the hydrogen atom Hamiltonian and perform DMC to obtain its
 ground-state energy and wavefunction.

8. *Hydrogen atom with a numpy Hamiltonian.

9. *Helium atom with regular and numpy Hamiltonians.

10. Write DMC with a drift term: importance-sampling DMC. Use a
 Pade-Jastrow trial wavefunction to obtain better convergence for the
 Helium atom.

The next few assignments aim to introduce Jones et al.'s Quantum Drude
Oscillator (QDO) model, using the xenon dimer as a model system.

11. Write the QDO Hamiltonian for a single xenon atom and sample its
ground-state energy, using regularized (erf(r)/r) electrostatics.

12. Write the QDO Hamiltonian for a xenon dimer and sample its
ground-state energy over a radial scan, using a two-term exponential
exchange.

Next we will develop the QDO water model.

13. *Implement the Hamiltonian for a QDO water dimer using parameters
from Jones et al., and plot the energy for radial and rotational scans
without DMC sampling.

14. *Use DMC sampling on radial and rotational scans of QDO water
 dimer.

15. Implement the QDO water dimer Hamiltonian using numpy, to improve
performance.

Next, we will develop VMC to improve the sampling.

16. Implement VMC and use it to find good initial parameters for the
harmonic oscillator. 

17. Use VMC on the Hydrogen atom. [1]

18. Use VMC on the Helim atom. [1]

19. Use VMC on the QDO water dimer.

Next, we will develop importance sampling DMC for the QDO water dimer.

20. Figure out a trial wavefunction that works for QDO I: Try Jastrow type

21. Trial wavefunction for QDO II: Understand the trial wavefunctions
of Jones et al.

22. Trial wavefunction for QDO III: Try the wavefunctions of Jones et
al.

At this point, we will have to see whether the results are converged
enough and decide whether to do a GPU implementation. If we don't need
a GPU implementation, the next step is to compare QDO water to
high-level quantum data.

23. Extend the implementation of QDO water to n-mers.

24. Compare QDO water trimers (and n-mers) to available QM data.

...

-- Implement on-site trial wavefunctions for neutral atom dimer
-- Compare to analytic results for neutral atom dimer

[1] http://www.physics.buffalo.edu/phy411-506-2008/chapter10/ch10-lec6.pdf​