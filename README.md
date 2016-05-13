# quantum-drude-oscillators
Coarse-grained model of electronic structure

## Introduction

Quantum drude oscillators[1][3][7] have recently been proposed as a
way to model many-body and multipolar electrostatics and dispersion.
The results[2][4] are nothing short of astounding: a water model fit
to only gas-phase properties (monomer polarizability and short-range
dimer energies) reproduces experimental results over a huge range of
conditions: the temperature of maximum density, hydrogen bonding
networks, properties of ice, liquid-vapor interfaces[5], and more[6].
 
Even better, the method is possible to simulate on Anton using
path-integral MD. There is a 100x slowdown due to simulation of
"copies" of the system, but the extra work is somewhat parallelizable.
 
Before we invest hardware/embedded-software design effort into this
new simulation method, we need to see for ourselves that it works as
well as advertised. A CPU or GPU implementation of PIMD is still a
significant amount of work, so it would be nice to have even cheaper
confirmation of the method's applicability to the problems we care
about.
 
Fortunately, there is a method called Diffusion Monte Carlo (DMC) that
can be used to compute energies of fixed configurations of molecules
for any bosonic Hamiltonian. The quantum drude oscillator hamiltonian
falls in this category, and in fact the developers of QDO use DMC to
fit parameters in their water model. It is a sampling method that
finds the ground-state wavefunction and energy for fixed nuclear
configurations by sampling electronic configurations.

## Intern project proposal

My proposal for an intern project is to build out DMC for QDO water to
the point where we can compute energies of dimers (2 water molecules),
and possibly up to 20-mers. With that, we can compare results to our
high-accuracy CCSD(T) quantum calculations and make the call on
whether to pursue this model further or not. The next step would be
parameterize a simple organic molecule like methane and compute
heterodimer energies (water-methane dimers) and compare those to
high-level QM calculations.
 
The water model is published in significant detail (see attached PDF
document) and so reproducing the results the authors claim is a simple
matter of implementation.
 
The next step, parameterizing methane, is a little more complicated
but the methodology is described in detail in Andrew Jones' thesis[3].

## Challenges

Sampling with DMC is computationally expensive but extremely
parallelizable. We will have to use GPUs and use a preprocessing step
called Variational Monte Carlo to create a trial/guide wavefunction
for importance-sampling DMC.
 
If the method is sufficiently accurate but too expensive to simulate
dynamics with, it can still be used to augment QM data generation for
our forcefield effort.

## What I've done so far

I have already (1 weekend and 6 evenings) written basic numpy code to
do DMC sampling of the QDO water model. My preliminary results are
promising (compare well to MP2) but too far from convergence to make a
solid case. A smart intern under my guidance could implement VMC,
implement importance-sampling DMC, implement a basic CUDA version of
DMC (2-5x speedup), compute results for QDO water clusters, and
parameterize methane.
 
This is incredibly important research and the finished code can be
given to Brent to expand the methodology beyond water and methane.
 
If the results are very good, we can invest effort in working on a GPU
version of PIMD to actually simulate dynamics using this model.
 
[1] http://journals.aps.org/prb/abstract/10.1103/PhysRevB.87.144103
[2] http://journals.aps.org/prl/abstract/10.1103/PhysRevLett.110.227801
[3] https://www.era.lib.ed.ac.uk/handle/1842/4878
[4] http://www.pnas.org/content/112/20/6341.full
[5] http://www-llb.cea.fr/WATSURF-2013/Presentations/ST/Flaviu_Cipcigan_les_Houches_2013.pdf
[6] http://www.ncbi.nlm.nih.gov/pubmed/16354045
[7] http://www.tandfonline.com/doi/abs/10.1080/00268976.2013.843032