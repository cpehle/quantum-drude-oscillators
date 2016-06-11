#!/usr/bin/env python

import numpy as np

if __name__ == "__main__":
    import argparse, subprocess, sys
    parser = argparse.ArgumentParser(description='Scan QDO radial')
    parser.add_argument('--upper', type=float, default=100, help='')
    parser.add_argument('--lower', type=float, default=0.1, help='')
    parser.add_argument('--nwalkers', type=int, default=10000, help='')
    parser.add_argument('--niters', type=int, default=9000, help='')
    parser.add_argument('--nconfigs', type=int, default=100, help='')

    args = parser.parse_args()

    for distance in np.linspace(args.lower, args.upper, args.nconfigs):
        print distance, " ",
        sys.stdout.flush()
        subprocess.check_call(["./dmc-xenon-dimer", str(args.niters), str(args.nwalkers),
                               str(distance)])
