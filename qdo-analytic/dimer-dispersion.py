#!/usr/bin/env python

import numpy as np

def monomer(omega, hbar=1.0):
    return (3./2) * hbar * omega

def alpha1(q, mu, omega):
    return q*q/(mu*omega*omega)

def alpha2(q, mu, omega, hbar=1.0):
    return (3./4) * (hbar/(mu*omega)) * alpha1(q, mu, omega)

def alpha3(q, mu, omega, hbar=1.0):
    return (5./4) * (hbar/(mu*omega))**2 * alpha1(q, mu, omega)

def C6(q, mu, omega, hbar=1.0):
    a1 = alpha1(q,mu,omega)
    return (3./4) * a1 * a1 * hbar * omega

def C8(q, mu, omega, hbar=1.0):
    a1 = alpha1(q, mu, omega)
    a2 = alpha2(q, mu, omega, hbar)
    return 5 * a1 * a2 * hbar * omega

def C10(q, mu, omega, hbar=1.0):
    c6 = C6(q, mu, omega, hbar)
    return (245./8) * (hbar/(mu*omega))**2 * c6

def dimer_dispersion(distances, q, mu, omega, terms=3, hbar=1.0):
    c6  = C6(q, mu, omega)
    c8  = C8(q, mu, omega, hbar)
    c10 = C10(q, mu, omega, hbar)
    answer = 2*monomer(omega, hbar)

    if terms >= 1:
        answer -= c6/xs**6
    if terms >= 2:
        answer -= c8/xs**8
    if terms >= 3:
        answer -= c10/xs**10

    return answer
    

if __name__ == "__main__":
    import argparse, subprocess, sys
    parser = argparse.ArgumentParser(description='Generate analytic QDO radial scan')
    parser.add_argument('--hbar', type=float, default=1.0, help="Reduced Planck's constant")
    parser.add_argument('--q', type=float, default=1.0, help='Drude charge')
    parser.add_argument('--mu', type=float, default=1.0, help='Drude mass')
    parser.add_argument('--omega', type=float, default=1.0, help='Ground-state monomer frequency')
    parser.add_argument('--lower', type=float, default=2.0, help='')
    parser.add_argument('--upper', type=float, default=10.0, help='')
    parser.add_argument('--nconfigs', type=int, default=100, help='')
    parser.add_argument('--terms', type=int, default=3, help='1, 2, or 3 terms in the series')

    args = parser.parse_args()

    xs = np.linspace(args.lower, args.upper, args.nconfigs)
    answer = dimer_dispersion(xs, args.q, args.mu, args.omega, args.terms, args.hbar)
    np.set_printoptions(precision=16)
    
    for r, e_r in zip(xs, answer):
        print r, e_r
