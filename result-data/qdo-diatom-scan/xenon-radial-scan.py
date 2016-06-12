#!/usr/bin/env python

import numpy as np

if __name__ == "__main__":
    import argparse, subprocess, sys
    parser = argparse.ArgumentParser(description='Scan QDO radial')
    parser.add_argument('--lower', type=float, default=5.0, help='')
    parser.add_argument('--upper', type=float, default=10.0, help='')
    parser.add_argument('--charge', type=float, default=2.0, help='')    
    parser.add_argument('--nwalkers', type=int, default=100000, help='')
    parser.add_argument('--niters', type=int, default=40000, help='')
    parser.add_argument('--nconfigs', type=int, default=10, help='')

    args = parser.parse_args()

    for distance in np.linspace(args.lower, args.upper, args.nconfigs):
        print distance,"",
        sys.stdout.flush()        
        subprocess.check_call(["./qdo-diatom", str(args.niters), str(args.nwalkers),
                               str(distance), str(args.charge)])
        sys.stdout.flush()
        
