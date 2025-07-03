#!/usr/bin/env python3
import os

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit

import unpack_trigger as trigger

import argparse

from calibrate import Calibrator

def compute_cumulative_rate(hist_tot, norm):    
    # scale counts to a rate:
    bins = np.arange(hist_tot.size)
    rate = hist_tot / norm
    err  = hist_tot**0.5 / norm

    cum_rate = np.cumsum(rate[::-1])[::-1]
    cum_err  = np.cumsum(err[::-1])[::-1]

    return bins, cum_rate, cum_err

def plot(hist, norm=0, ax=None, **kwargs): 

    cbins = np.arange(hist.size)
    rate = hist.astype(float)

    cbins = cbins[1:]
    rate = np.sum(rate) - np.cumsum(rate)[:-1]

    err = np.sqrt(rate)

    # scale counts to a rate:
    if norm:
        rate /= norm
        err /= norm
        plt.ylabel("Rate (triggers per image)")
    else:
        plt.ylabel('Counts') 

    if ax:
        ax.errorbar(cbins,rate,yerr=err,fmt="o", **kwargs)
    else:
        plt.errorbar(cbins,rate,yerr=err,fmt="o", **kwargs)

    #plt.savefig("plots/rate.pdf")

def process_trig(filename,calibrator,verbose=False,hot=False):
    # first unpack and display file contents
    header,px,py,highest,region,timestamp,millistamp,images,dropped,millis_images = trigger.unpack_all(filename)
    #^ this line takes soooooooo much time
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
    keep1 = (highest > 0) | (rcenter < min(threshold))

    # remove hot pixels
    if hot:
        f_name = os.path.join(args.calib, 'hot_offline.npz')
        f_hot  = np.load(f_name)
        hot_pixels = f_hot['hot_list']
        pixels = px + 5328 * py
        keep2 = np.logical_not(np.isin(pixels, hot_pixels))
    else:
        keep2 = True

    keep = keep1 & keep2
        
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
    

def calibrate_thresholds(rate, n_trig):
    cum_rate = np.sum(rate) - np.cumsum(rate)[:-1]
    prescale = 1
    for i in np.arange(cum_rate.size,0,-1)-1:
        while cum_rate[i] >= n_trig*prescale:
            print("prescale: ", prescale, "threshold: ", i+2)
            prescale *= 8
    


if __name__ == "__main__":
    example_text = '''examples:

    ...'''
    
    parser = argparse.ArgumentParser(description='Plot rate from Cosmics.', epilog=example_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', nargs='+', help='files to process')
    parser.add_argument('--sandbox',action="store_true", help="run sandbox code and exit (for development).")
    parser.add_argument('--max',  type=int, default=1025,help="maximum pixel value in rate plot (x-axis).")
    parser.add_argument('--n_trig',  metavar='NUM', type=int, default=0,help="find prescales and thresholds yielding NUM pixels per event.")
    parser.add_argument('--small', action='store_true', help='Generate a small plot')
    parser.add_argument('-v', '--verbose', action='store_true', help='enable verbose output')
    parser.add_argument('--calib',default='calib', help="path to calibration files")
    parser.add_argument('--efficiency', action='store_true', help='plot efficiency vs rate')
    parser.add_argument('--electrons', action='store_true', help='plot in terms of number of electrons')
    parser.add_argument('--hot', action='store_true', help='ignore pixels in calib/hot_offline.npz')
    args = parser.parse_args()

    if args.efficiency and not args.electrons:
        raise ValueError("must use '--electrons' to use '--efficiency'")
        

    if args.electrons:
        # to convert pixel value to number of electrons
        lens = np.load(os.path.join(args.calib, 'lens.npz'))
        try:
            parameters = lens["secant_parameters"]
            factor = parameters[1]
            xlabel = "Threshold (Number of electrons)"
        except:
            print("no value of K0 found in lens.npz, use --radial option when finding lens shading")
            print("making units of threshold pixel value")
            factor = 1
            xlabel = "Threshold (Pixel value)"
    else:
        factor = 1
        xlabel = "Threshold (Pixel value)"

    calibrator = Calibrator(args.calib)
    
    hist_trig = 0
    norm_trig = 0
    thresholds = None
    prescales = None

    for filename in args.files:
        if args.verbose:
            print("processing trigger file:", filename)
        h, norm, th, ps = process_trig(filename, calibrator, args.verbose, args.hot)
        #^ this take a lot of time, why does it take so much time? 
        if not thresholds is None and not np.all(thresholds == th):
            raise ValueError('Non-matching triggers found.')
        thresholds = th
        prescales = ps

        hist_trig += h
        norm_trig += norm



    bins, rate, err = compute_cumulative_rate(hist_trig,norm_trig)
    bins = bins / factor

    plt.plot(bins,rate,'b', label="Trigger rate")
    plt.fill_between(bins,rate+err,rate-err, color='b', alpha = .5)

    plt.xlabel(xlabel)
    plt.ylabel("Rate per Image")
    plt.semilogy()
    plt.xlim(0,args.max/factor)
    #print(rate[0])
    #print(hist_trig[0])

    crm_rate = 0.00307/2 #rate per second/2 frames per second
    # cosmic ray muon rate for samsung galxy s6 camera
    # flux taken from A. Dragic, et al. 2007
    # Measurement of cosmic ray muon flux in the Belgrade ground level and underground laboratories

    plt.plot([0,args.max/factor], [crm_rate,crm_rate], 'k', label="Muon Rate")
    
    plt.legend()
    plt.show()

    
    if args.efficiency:
        x = np.array([43, 50, 100])
        y = np.array([.74, .72, .47])
        yerr = np.array([.07, .06, .04])
        #^ taken from mikes paper on efficiency

        def lin_func(x, m, b):
            return m*x+b

        
        
        a,cov=curve_fit(lin_func,x,y,sigma=yerr,absolute_sigma=True)

        real_bins = a[0]*bins + a[1]
        
        plt.plot(real_bins, rate, 'm', label = "Rate of Triggers")
        plt.fill_between(real_bins,rate+err,rate-err, color='m', alpha = .5)
 
        x_arr = np.linspace(.4, .8)
        mdr_arr = x_arr*crm_rate
        plt.plot(x_arr, mdr_arr, 'k', label="Muon Detection Rate")
        plt.xlim(.45,.75)
        plt.ylim(0.0001,1)
        plt.semilogy()
        plt.xlabel('efficiency')
        plt.ylabel('rate per image')
        plt.legend()
        plt.show()

    if args.n_trig:
        calibrate_thresholds(hist_wgt/images, args.n_trig)

