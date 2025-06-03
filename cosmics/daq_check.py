#!/usr/bin/env python3

import sys
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import unpack_hist as hist
import unpack_trigger as trigger

from calibrate import Calibrator

def process_hist(filename, raw=False):

    header,hist_cln,hist_hot,hist_wgt = hist.unpack_all(filename)

    images = hist.interpret_header(header, "images")
    prescale = hist.interpret_header(header, "hist_prescale")
        
    hist_tot = hist_cln.astype(float) if raw else hist_wgt.astype(float)
    
    return hist_tot, images / prescale


def compute_rate(hist_tot, norm):    
    # scale counts to a rate:
    bins = np.arange(hist_tot.size)
    rate = hist_tot / norm
    err  = hist_tot**0.5 / norm

    return bins, rate, err

def process_trig(filename,calibrator,verbose=False):
    # first unpack and display file contents
    header,px,py,highest,region,timestamp,millistamp,images,dropped,millis_images = trigger.unpack_all(filename)
    
    threshold,prescale = trigger.get_trigger(header) 
    
    # sort thresholds from lowest to highest
    argsort = np.argsort(threshold)
    threshold = threshold[argsort]
    prescale = prescale[argsort]
    
    nzb    = trigger.interpret_header(header, 'num_zerobias') 
    width  = trigger.interpret_header(header, 'width')
    height = trigger.interpret_header(header, 'height')

    if verbose:
        trigger.show_header(header)
        print("zero-bias:  ", np.sum(highest==0)/images)
        for i in range(prescale.size):
            print("{} prescale ({}):  {}".format(i, prescale[i], np.sum(highest==i+1)/images))


    if calibrator:
        region = calibrator.calibrate_region(px,py,region,header)

    # get triggered pixel values
    rcenter = region[:, region.shape[1]//2]
    
    # remove zero bias triggers above lowest threshold for easy plotting
    keep = (highest > 0) | (rcenter < min(threshold))
    px = px[keep]
    py = py[keep]
    highest = highest[keep]
    rcenter = rcenter[keep]

    hist, _ = np.histogram(rcenter, bins=np.arange(1025))

    # now calculate normalizations
    th_adj = np.hstack([[0], threshold])
    ps_adj = np.hstack([[width*height/nzb], prescale])

    norm_ps = images / ps_adj
    max_thresh_idx = np.argmin(np.arange(1024) >= th_adj.reshape(-1,1), axis=0)
    max_thresh_idx[max_thresh_idx == 0] = len(th_adj)
    norm = norm_ps[max_thresh_idx-1]

    return hist, norm, th_adj, ps_adj

if __name__ == "__main__":
    example_text = '''examples:

    ...'''
    
    parser = argparse.ArgumentParser(description='Plot rate from Cosmics.', epilog=example_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--hist', metavar='HIST', required=True, nargs='+', help='histogram file(s) to process')
    parser.add_argument('--trig', metavar='TRIG', required=True, nargs='+', help='trigger file(s) to process')
    parser.add_argument('--sandbox',action="store_true", help="run sandbox code and exit (for development).")
    parser.add_argument('--calib',default='calib', help="path to calibration files")
    parser.add_argument('-r', '--raw', action='store_true', help='use unweighted values')
    parser.add_argument('--max',  type=int, default=1024,help="maximum pixel value in rate plot (x-axis).")
    parser.add_argument('-v', '--verbose', action='store_true', help='enable verbose output')
    parser.add_argument('--electrons', action='store_true', help='plot in terms of number of electrons')
    
    args = parser.parse_args()

    if args.electrons:
        # to convert pixel value to number of electrons
        lens = np.load(os.path.join(args.calib, 'lens.npz'))
        try:
            parameters = lens["secant_parameters"]
            factor = parameters[1]
            xlabel = "Threshold ($e^{-}$)"
        except:
            print("no value of K0 found in lens.npz, use --radial option when finding lens shading")
            print("making units of threshold pixel value")
            factor = 1
            xlabel = "Threshold (pixel value)"

    else:
        factor = 1
        xlabel = "Threshold (pixel value)"
        

    hist_tot = 0
    img_tot  = 0

    for filename in args.hist:
        if args.verbose:
            print('processing hist file:', filename)
        h, img = process_hist(filename, raw=args.raw)
        hist_tot += h
        img_tot  += img

    hist_bins, hist_rate, hist_err = compute_rate(hist_tot, img_tot)

    calibrator = Calibrator(args.calib) if not args.raw else None
    
    
    hist_trig = 0
    norm_trig = 0
    thresholds = None
    prescales = None

    for filename in args.trig:
        if args.verbose:
            print("processing trigger file:", filename)
        h, norm, th, ps = process_trig(filename, calibrator, args.verbose)
        if not thresholds is None and not np.all(thresholds == th):
            raise ValueError('Non-matching triggers found.')
        thresholds = th
        prescales = ps

        hist_trig += h
        norm_trig += norm

    trig_bins, trig_rate, trig_err = compute_rate(hist_trig, norm_trig)

    # now create plot
    plt.errorbar(hist_bins/factor,hist_rate,yerr=hist_err,color="black",fmt="--", label='histogram')

    for i in range(len(thresholds)):
        label = 'prescale: {}'.format(prescales[i]) \
                if thresholds[i] else 'zero-bias'

        th_min = thresholds[i]
        th_max = thresholds[i+1] if i<len(thresholds)-1 else 1024

        bins = trig_bins[th_min:th_max] / factor
        # factor = 1 when units of bins is pixel value 
        rate = trig_rate[th_min:th_max]
        err  = trig_err[th_min:th_max]
        plt.errorbar(bins,rate,yerr=err,fmt="o", label=label)

    plt.xlabel(xlabel)
    plt.ylabel("rate per image")
    plt.semilogy()
    plt.xlim(0,args.max/factor)

    plt.legend()
    plt.show()


