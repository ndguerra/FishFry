#!/usr/bin/env python3

import sys
import os
from unpack import *
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from geometry import load_res
from dark_pixels import load_dark
from lens_shading import load_weights

import argparse

def process(filename, args):

    # load data, from either raw file directly from phone, or as output from combine.py utility:

    if args.raw:
         version,header,sum,ssq,max,second = unpack_all(filename)
         index = get_pixel_indices(header)
         images = interpret_header(header, "images")
         num = np.full(index.size, images)
         width  = interpret_header(header, "width")
         height = interpret_header(header, "height")
    else:
        # load the image geometry from the calibrations:
        width, height = load_res(args.calib)

        try:
            npz = np.load(filename)
        except:
            print("could not process file ", filename, " as .npz file.  Use --raw option?")
            return

        sum     = npz['sum']
        ssq     = npz['ssq']
        num     = npz['num'] 
    
    cmean = sum / num
    cvari = (ssq / num - cmean**2) * num / (num-1)

    # apply gains if appropriate
    if args.gain:
        try:
            wgt = load_weights(args.calib).flatten()
        except IOError:
            print("weights not found.")
            return
         
        cmean = cmean * wgt
        cvari = cvari * wgt**2

    # select pixels to plot
    index = np.arange(sum.size)
    xpos = index % width
    ypos = index // width
    rpos = np.sqrt((xpos - xpos.mean())**2 + (ypos - ypos.mean())**2)
    keep = np.ones(width*height, dtype=bool)
    keep &= (num != 0)
    
    if args.no_dark or args.all_dark:
        try:
            dark = load_dark(args.calib)
        except IOError:
            print("dark pixel file does not exist.")
            return
        
        if args.no_dark:
            keep &= np.logical_not(dark)
        if args.all_dark:
            keep &= dark


    # first show spatial distribution

    if args.by_radius:
        plt.figure(2, figsize=(6,8))
        plt.subplot(211)
        plt.hist2d(rpos[keep], cmean[keep],norm=LogNorm(),bins=[500,500],range=[[0,rpos.max()],[0,args.max_mean]], cmap='seismic')
        plt.xlabel('radius')
        plt.ylabel('mean')

        plt.subplot(212)
        plt.hist2d(rpos[keep], cvari[keep],norm=LogNorm(),bins=[500,500],range=[[0,rpos.max()],[0,args.max_var]], cmap='seismic')
        plt.xlabel('radius')
        plt.ylabel('variance')

    if args.spatial_plots:
        plt.figure(3, figsize=(6,8))
        plt.subplot(211)
        plt.imshow(cmean.reshape(height, width), 
                cmap='seismic', vmax=args.max_mean)
        plt.colorbar()
        plt.title("mean")

        plt.subplot(212)
        plt.imshow(cvari.reshape(height, width), #norm=LogNorm(), 
                cmap='seismic', vmax=args.max_var)
        plt.colorbar()
        plt.title("variance")

    # now do 2D histogram(s) for mean and variance 
             
    plt.figure(1, figsize=(10,8))
    if args.by_filter:
        # 4 subplots 

        for i in range(4):
            ix = i % 2
            iy = i // 2

            pos = keep * ((xpos%2)==ix) * ((ypos%2)==iy)
        
            plt.subplot(2,2,i+1)
            plt.hist2d(cmean[pos],cvari[pos],norm=LogNorm(), bins=(500,500), range=((0,args.max_mean),(0, args.max_var)))
            plt.xlabel("mean")
            plt.ylabel("variance")    

    else:
        plt.hist2d(cmean[keep],cvari[keep],norm=LogNorm(),bins=[500,500],range=[[0,args.max_mean],[0,args.max_var]])
        plt.xlabel("mean")
        plt.ylabel("variance")
        

    

    if args.save_plot:
        plot_name = "plots/mean_var_calib.pdf" if args.calib \
                else "plots/mean_var.pdf"
        print("saving plot to file:  ", plot_name)
        plt.savefig(plot_name)
        
    plt.show()

    if args.hists:
        h,bins = np.histogram(np.clip(cmean,0,10), bins=100, range=(0,10))
        err = h**0.5
        cbins = 0.5*(bins[:-1] + bins[1:])
        plt.errorbar(cbins,h,yerr=err,color="black",fmt="o")
        plt.ylim(1.0,1E7)
        plt.ylabel("pixels")
        plt.xlabel("mean")
        plt.yscale('log')
        plt.savefig("hmean.pdf")
        plt.show()

        h,bins = np.histogram(np.clip(cvari,0,100), bins=100, range=(0,100))
        err = h**0.5
        cbins = 0.5*(bins[:-1] + bins[1:])
        plt.errorbar(cbins,h,yerr=err,color="black",fmt="o")
        plt.ylim(1.0,1E7)
        plt.ylabel("pixels")
        plt.xlabel("variance")
        plt.yscale('log')
        plt.savefig("hvari.pdf")
        plt.show()

    return 
    
if __name__ == "__main__":
    example_text = '''examples:

    ./show_mean_var.py ./data/combined/small_dark.npz --max_var=30 --max_mean=5'''
    
    parser = argparse.ArgumentParser(description='Plot mean and variance.', epilog=example_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', metavar='FILE', nargs='+', help='file to process')
    parser.add_argument('--sandbox',action="store_true", help="run sandbox code and exit (for development).")
    parser.add_argument('--raw',action="store_true", help="input files have not been preprocessed.")
    parser.add_argument('--max_var',  type=float, default=3000,help="variance limit for plots")
    parser.add_argument('--max_mean', type=float, default=1023,help="mean limit for plots")
    parser.add_argument('--no_dark',action="store_true", help="drop dark pixels from all plots.")
    # fix ^this 
    parser.add_argument('--all_dark',action="store_true", help="drop non-dark pixels from all plots.")
    # fix ^this
    parser.add_argument('--by_filter',action="store_true", help="produce 4 plots for each corner of the 2x2 filter arrangement.")
    # don't know why we have ^this 
    parser.add_argument('--by_radius',action="store_true", help="make 2D histograms of mean or variance and radius")
    # ^this isn't very insightful
    parser.add_argument('--gain',action="store_true", help="apply gain correction.")
    parser.add_argument('--calib', default='calib', help='directory with calibration files')
    parser.add_argument('--save_plot', action='store_true', help='Save plots in ./plots/ directory')
    parser.add_argument('--spatial_plots', action='store_true', help='show spatial plots of mean and variance')
    parser.add_argument('--hists', action='store_true', help='show histograms of mean and variance')
    args = parser.parse_args()
    


    if args.sandbox:
        print('========================================')
        print('-- in sandbox development environment --')
        print('========================================')
        print()

    for filename in args.files:
        print("processing file:  ", filename)
        process(filename, args)




