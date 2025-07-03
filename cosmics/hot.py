#!/usr/bin/env python3

# dump a header from run data

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
#import ROOT as r

from unpack_trigger import unpack_all, show_header, interpret_header
from calibrate import Calibrator

width = 0


def process_dat(filename, calibrator, thresh=0, verbose=False):
    global width
    header,px,py,highest,region,timestamp,millistamp,images,dropped,millis_images = unpack_all(filename)

    if not width: # dont use files with different widths
        width = interpret_header(header, "width")
    
    if verbose:
        show_header(header)

    if calibrator:
        region = calibrator.calibrate_region(px,py,region,header)


    icenter = region.shape[1] // 2

    # if thresh is set, use it to count occupancies
    # otherwise, use the the highest prescale
    idx_regions = (highest == highest.max())
    if thresh: 
        idx_regions &= (region[:, icenter] >= thresh)

    # return flattened indices with hits
    idx_occ = py[idx_regions]*width + px[idx_regions]

    
    return np.histogram(idx_occ, bins=np.arange(total_pixels+1))[0]

 
def process_root(filename, calibrator, thresh=0, verbose=False):
    f = r.TFile(filename)
    trig = f.Get('triggers')

    histogram = np.zeros(calibrator.width*calibrator.height)

    for ievt, evt in enumerate(trig):
        if verbose:
            print(ievt+1, '/', trig.GetEntries(), end='\r')
        for x,y,cal in zip(evt.x, evt.y, evt.cal):
        
            if cal >= thresh:
                idx = y * calibrator.width + x
                if idx in calibrator.hot: continue
                histogram[idx] += 1

    return histogram

    
if __name__ == "__main__":

    example_text = ""
    
    parser = argparse.ArgumentParser(description='Offline (second pass) hot pixel finding.', epilog=example_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', metavar='FILE', nargs='+', help='file to process')
    parser.add_argument('--sandbox',action="store_true", help="run trial code")
    parser.add_argument('--thresh',  type=int, default=0, help="calibrated threshold for occupancy count")
    parser.add_argument('--calib', default='calib',help='location of calibration directory')
    parser.add_argument('--maxocc', default=1, type=int, help='maximum number of hits for "clean" pixel')
    parser.add_argument('-p', '--plot', action='store_true', help='plot pixel occupancies')
    parser.add_argument('-s', '--small', action='store_true', help='Decrease plot size')
    parser.add_argument('-c', '--commit',action="store_true", help="save hot pixels to file.")
    parser.add_argument('-o', '--offline', action='store_true', help='add to the offline hot pixels (instead of overwriting)')
    parser.add_argument('-v', '--verbose', action="store_true", help="display file summary")
    parser.add_argument('-r', '--raw', action="store_true", help="Use raw values")

    args = parser.parse_args()

    calibrator = Calibrator(args.calib, offline=args.offline)
    total_pixels = calibrator.width * calibrator.height
    print('total pixels:', total_pixels)
    occ = np.zeros(total_pixels)

    if args.raw:
        calibrator = None

    end = "\n"# if args.verbose else "\r"
    for filename in args.files:
        print("processing file:  ", filename, end=end)
        if filename.endswith('.dat'):
            occ += process_dat(filename, 
                calibrator, 
                args.thresh, 
                verbose=args.verbose)
        elif filename.endswith('.root'):
            occ += process_root(filename,
                calibrator,
                args.thresh,
                verbose=args.verbose)
        else:
            print('Skipping file', filename)


    print("max occupancy:  ", np.max(occ))
    print("total hits:     ", np.sum(occ))
    print("single hits:    ", np.sum(occ == 1))
    print("hot pixels:     ", np.sum(occ > args.maxocc))
    print("hot events:     ", np.sum(occ[occ > args.maxocc]))

    
    hot = np.argwhere(occ > args.maxocc).flatten()
    
    if args.plot:
        figsize = (4,3.2) if args.small else (7,5)
        plt.figure(figsize=figsize, tight_layout=True)
        if args.maxocc > 1:
            plt.hist(occ, bins=np.arange(args.maxocc+1), log=True, histtype='stepfilled')
        elif occ.max() <= 200: 
            plt.hist(occ, bins=np.arange(occ.max()+1), log=True, histtype='stepfilled')
        else:
            plt.hist(occ, bins=200, log=True)
        #bins = np.arange(args.maxocc+1)
        #occ_hist = np.histogram(occ, bins=bins)[0]
        #plt.hist(bins[:-1]/50000, bins=bins/50000, weights=occ_hist, log=True, histtype='stepfilled')
        
        #plt.errorbar(bins[1:-1], occ_hist[1:], yerr=np.sqrt(occ_hist[1:]), color='k')
        #plt.loglog()
        #plt.xlabel('Frequency of pixel hits [frames$^{-1}$]')
        plt.xlabel('Number of triggers in 50,000 frames')
        plt.ylabel('Number of pixels')
        plt.show()

    if args.commit:
        print("saving ", hot.size, " hot pixels to file.")
        fname = os.path.join(args.calib, 'hot_offline.npz')
        if args.offline:
            fhot = np.load(fname)
            hot = np.hstack([hot, fhot['hot_list']])
        np.savez(fname, hot_list=hot)

