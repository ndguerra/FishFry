#!/usr/bin/env python3

import argparse
import sys
import os

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from unpack import *
from geometry import load_res
from dark_pixels import load_dark


def load_weights(calib):
    f_lens = np.load(os.path.join(calib, 'lens.npz'))
    
    wgt  = f_lens['wgt']
    ds   = f_lens['down_sample']

    f_lens.close()

    wgt[np.isnan(wgt) | np.isinf(wgt) | (wgt < 0)] = 0
    wgt_all = np.repeat(np.repeat(wgt, ds, axis=0), ds, axis=1)
        
    return wgt_all


def downsample(calib, rsq_thresh=0, calib_dark=True, calib_hot=True, ds=4):

    print('loading image geometry')
    width, height = load_res(calib)
    index  = np.arange(width*height,dtype=int)
    xpos = index % width
    ypos = index / width

    # load the pixel gains:
    try:
        print('loading pixel gains')
        filename = os.path.join(calib, "gain.npz")
        fgains  = np.load(filename)
    except:
        print("could not process file ", filename, " as .npz file.  Run gain.py with --commit option first?")
        return
    gain       = fgains['gain']
    intercept  = fgains['intercept'] 
    rsq        = fgains['rsq']

    fgains.close()

    # set bad pixels to NaN
    infs = np.isinf(gain) | np.isinf(intercept) | np.isnan(intercept)
    gain[infs] = np.nan
    intercept[infs] = np.nan

    if rsq_thresh:
        gain[rsq < rsq_thresh] = np.nan
        intercept[rsq < rsq_thresh] = np.nan

    if calib_dark:
        try:
            print('loading dark pixels')
            dark = load_dark(calib)
            gain[dark] = np.nan
            intercept[dark] = np.nan
        except IOError:
            print("dark pixel file could not be processed")
     
    if calib_hot:
        from hot_pixels import load_hot
        try:
            print('loading hot pixels')
            hot = load_hot(calib)
            gain[hot] = np.nan
            intercept[hot] = np.nan
        except IOError:
            print("hot pixel file could not be processed") 

    print('beginning to downsample')

    nx = width // ds
    ny = height // ds

    # take mean of each n x n block
    gain_ds = gain.reshape(ny, ds, nx, ds)
    intercept_ds = intercept.reshape(ny, ds, nx, ds)
    g_av = np.nanmedian(gain_ds, axis=(1,3))
    b_av = np.nanmedian(intercept_ds, axis=(1,3))
    
    print('downsampling completed, displaying information:')
    print('nx:                ', nx)
    print('ny:                ', ny)
    print('lens.shape:        ', g_av.shape)
    print('NaN found:         ', np.sum(np.isnan(g_av)))
    print()
    
    print('computed lens shading attributes')

    return g_av, b_av

def radial_correct(stat, n_points=20, plot=False, small=False):
    sy, sx = stat.shape
    
    X, Y = np.meshgrid(np.arange(sx), np.arange(sy))
    R = np.sqrt((X-X.mean())**2 + (Y-Y.mean())**2) 

    fit_x = np.linspace(0, R.max(), n_points) 

    # use a piecewise linear fit
    def make_segment(xi, xf, yi, yf):
        def _(x):
            return yi + (x - xi)/(xf-xi)*(yf-yi)
        return _
    
    # piecewise linear fit
    def f(x, *y):
        funclist = [make_segment(*xy) for xy in zip(fit_x[:-1], fit_x[1:], y[:-1], y[1:])]
        return np.piecewise(x, [x >= xi for xi in fit_x[:-1]], funclist)

    p0 = [2 + i**2/100 for i in range(n_points)]
    y, _ = optimize.curve_fit(f, R.flatten(), stat.flatten(), p0)
    fit_y = f(fit_x, *y)

    # naive sec^(2p) function
    def g(x, x0, a):
        return a*(1 + (x/x0)**2)**2

    p1 = [1000, np.mean(stat)]
    p, _ = optimize.curve_fit(g, R.flatten(), stat.flatten(), p1)
    fit_sec = g(fit_x, *p)

    print('Secant fit:')
    print('a = {:.3f}'.format(p[1]))
    #print('p = {:.3f}'.format(p[2]/2))
    print('z = {:.3f}'.format(p[0]))

    if plot:
        figsize = (3.5,2.6) if small else (7,5)
        plt.rc('font', size=16)
        plt.figure(figsize=figsize, tight_layout=True)
        plt.plot(fit_x*4, fit_y, 'y-', label='Piecewise linear')
        plt.plot(fit_x*4, fit_sec, 'r-', label=r'$\sec^4 \theta$')
        plt.hist2d(R.flatten()*4, stat.flatten(), bins=(500,500), norm=LogNorm(), cmap='Purples')
        #plt.title('Radial gain fit')
        plt.xlabel('Radius [pixels]')
        plt.ylabel('Gain')
        plt.legend()

    return np.interp(R, fit_x, fit_y).reshape(sy, sx), p


def load(calib):    
    flens = np.load(os.path.join(calib, 'lens.npz'))
    #lens  = flens["lens"]
    #flens.close()
    return flens

def plot(lens, min_gain=None, max_gain=None, small=False, K0 = None):
    print('plotting computed lens')
    figsize = (4, 2.6) if small else (8, 5)
    plt.figure(figsize=figsize, tight_layout=True)
    plt.rc('font', size=16)        
    plt.imshow(lens, vmin=min_gain, vmax=max_gain, origin='lower',
            extent=[0, lens.shape[1]*4, 0, lens.shape[0]*4])
    plt.title("$K(x,y)$")
    plt.xlabel('x [pixels]')
    plt.ylabel('y [pixels]')
    plt.colorbar()
    #plt.savefig("plots/lens.pdf")
    plt.show()
     

        
if __name__ == "__main__":
    example_text = '''examples:

    ...'''
    
    parser = argparse.ArgumentParser(description='construct the lens shading map from calculated gains', epilog=example_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sandbox',action="store_true", help="run sandbox code and exit (for development).")
    parser.add_argument('--calib',default='calib', help='calibration directory to use')
    parser.add_argument('--keep_dark',action="store_true", help="explicitly include dark pixels in analysis.")
    parser.add_argument('--keep_hot',action="store_true", help="explicitly include hot pixels in analysis.")
    parser.add_argument('--min_rsq',type=float,help='minimum r^2 value to be considered in downsampled gain')
    parser.add_argument('--down_sample',  metavar="DOWN", type=int, default=4,help="down sample by amount DOWN.")
    parser.add_argument('--max_gain',  type=float, help="minimum gain for plot.")
    parser.add_argument('--min_gain',  type=float, help="maximum gain for plot.")
    parser.add_argument('--plot_only',action="store_true", help="load and plot previous results.")
    parser.add_argument('--radial', action='store_true', help='Use radially symmetric weights')
    parser.add_argument('--commit', action="store_true", help="commit lens.npz file")
    parser.add_argument('--plot', action="store_true", help="plot lens shading values")
    parser.add_argument('--small', action='store_true', help='generate smaller plots')

    args = parser.parse_args()
    
    if args.plot_only:
        lens = load(args.calib)
        try:
            K0 = lens['secant_parameters'][1]
        except:
            K0 = None
        plot(1/lens['wgt'], args.min_gain, args.max_gain, small=args.small, K0 = K0)
        
    else: 
        lens, offset = downsample(args.calib, 
                calib_dark=(not args.keep_dark), 
                calib_hot=(not args.keep_hot), 
                rsq_thresh = args.min_rsq, 
                ds=args.down_sample)
  
        if args.radial:
            lens, p = radial_correct(lens, plot=args.plot, small=args.small)
        else:
            p = None
            
        if args.commit:
            
            filename = os.path.join(args.calib, 'lens.npz')
            print('computed lens shading committed to ', filename)
            if args.radial:
                np.savez(filename,
                    down_sample=args.down_sample, 
                    wgt=lens.min()/lens,
                    secant_parameters = p)
            else:
                np.savez(filename, 
                    down_sample=args.down_sample, 
                    wgt=lens.min()/lens)

        if args.plot:
            plot(lens, args.min_gain, args.max_gain, args.small, p[1])


