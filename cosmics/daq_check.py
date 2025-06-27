#!/usr/bin/env python3

import sys
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
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
    parser.add_argument('--efficiency', action='store_true', help='plot efficiency vs rate')
    
    args = parser.parse_args()

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
        #^ this take a lot of time, why does it take so much time? 
        if not thresholds is None and not np.all(thresholds == th):
            raise ValueError('Non-matching triggers found.')
        thresholds = th
        prescales = ps

        hist_trig += h
        norm_trig += norm

    trig_bins, trig_rate, trig_err = compute_rate(hist_trig, norm_trig)
    
    real_rate = np.cumsum(hist_rate[::-1])[::-1]
    real_err  = np.cumsum(hist_err[::-1])[::-1]
    
    # now create plot
    #print(np.sum(hist_rate))
    #plt.errorbar(hist_bins/factor,real_rate,yerr=hist_err,color="black",fmt="--", label='histogram', zorder=0) # rates as a function of pixel value
    #plt.plot(hist_bins/factor, real_rate, 'r')
    #plt.fill_between(hist_bins/factor,real_rate+real_err,real_rate-real_err, color='r', alpha = .5) # rates as a function of threshold


    real_bins = np.array([])
    real_rate = np.array([])
    real_err  = np.array([])

    #trig_rate[762] = 0
    #trig_err[762] = 0
    #print(trig_rate[805])
    #trig_rate[805] = 0
    #trig_err[805] = 0
    
    
    value1 = (trig_bins > 370)
    value2 = (trig_rate > 0.01)
    value3 = value1 & value2
    trig_rate[value3] = 0
    trig_err[value3] = 0
    print(np.where(value3))
    
    for i in range(len(thresholds)-1,0,-1): #! backwards
        label = 'prescale: {}'.format(prescales[i]) \
                if thresholds[i] else 'zero-bias'

        th_min = thresholds[i]
        th_max = thresholds[i+1] if i<len(thresholds)-1 else 1024

        bins = trig_bins[th_min:th_max] / factor
        # factor equals 1 when units of bins is pixel value 
        rate = trig_rate[th_min:th_max]
        err  = trig_err[th_min:th_max]

        
        real_bins = np.append(bins, real_bins)
        count = real_rate[0] if np.size(real_rate) else 0
        real_rate = np.append(np.cumsum(rate[::-1])[::-1] + count, real_rate)
        real_err  = np.append(np.cumsum(err[::-1])[::-1], real_err)
    
        #plt.errorbar(bins,rate,yerr=err,fmt="o", label=label, zorder=0) # first plot
    plt.plot(real_bins, real_rate, 'b-')
    plt.fill_between(real_bins,real_rate+real_err,real_rate-real_err, color='b', alpha = .5, label = "Rate of triggers")

    plt.xlabel(xlabel)
    plt.ylabel("rate per image")
    plt.semilogy()
    plt.xlim(0,args.max/factor)

    crm_rate = 0.00307/2 #rate per second/2 frames per second
    crm_err  = 0.00013/2 
    # cosmic ray muon rate for samsung galxy s6 camera
    # flux taken from A. Dragic, et al. 2007
    # Measurement of cosmic ray muon flux in the Belgrade ground level and underground laboratories

    plt.plot(real_bins-4, np.ones(np.size(real_bins))*crm_rate, 'k', label="Muon Rate", zorder=1)
    #plt.fill_between(hist_bins/factor, crm_rate-crm_err, crm_rate+crm_err) 
    
    plt.legend()
    plt.show()

    plt.close()

    if args.efficiency:
        x = np.array([43, 50, 100])
        y = np.array([.74, .72, .47])
        yerr = np.array([.07, .06, .04])
        #^ taken from mikes paper on efficiency

        def lin_func(x, m, b):
            return m*x+b

        
        
        a,cov=curve_fit(lin_func,x,y,sigma=yerr,absolute_sigma=True)

        real_bins = a[0]*real_bins + a[1]
        
        plt.plot(real_bins, real_rate, 'm', label = "Rate of Triggers")
        plt.fill_between(real_bins,real_rate+real_err,real_rate-real_err, color='m', alpha = .5)
 
        x_arr = np.linspace(.4, .8)
        mdr_arr = x_arr*crm_rate
        plt.plot(x_arr, mdr_arr, 'k', label="Muon Detection Rate")
        #plt.fill_between(x_arr, x_arr/(crm_rate+crm_err), x_arr/(crm_rate-crm_err),alpha = .5)

        x1 = real_rate
        y1 = a[0]*real_bins+a[1]
        y2 = cov[0,0]*real_bins+cov[1,1]
        #plt.plot(x1, y1, 'm')
        #plt.fill_between(x1,y1+y2,y1-y2, color='m', alpha = .5)

        #plt.errorbar(real_rate, a[0]*real_bins+a[1], yerr=cov[0,0]*real_bins+cov[1,1],\
        #             xerr=real_err, fmt='bo', label="Data points")
        plt.xlim(.45,.75)
        plt.ylim(0.0001,1)
        plt.semilogy()
        plt.xlabel('efficiency')
        plt.ylabel('rate per image')
        plt.legend()
        plt.show()


