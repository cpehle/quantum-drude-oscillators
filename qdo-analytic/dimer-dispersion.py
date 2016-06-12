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

    args = parser.parse_args()

    c6  = C6(args.q, args.mu, args.omega)
    c8  = C8(args.q, args.mu, args.omega, args.hbar)
    c10 = C10(args.q, args.mu, args.omega, args.hbar)

    xs = np.linspace(args.lower, args.upper, args.nconfigs)

    np.set_printoptions(precision=16)
    answer = 2*monomer(args.omega, args.hbar) - c6/xs**6 - c8/xs**8 - c10/xs**10

    for r, e_r in zip(xs, answer):
        print r, e_r
