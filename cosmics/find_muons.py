#!/usr/bin/env python3

import sys
import os
import argparse
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from unpack_trigger import unpack_all, get_trigger
from calibrate import Calibrator

#import ROOT as r

kernel5 = np.array([[0,0,0,0,0],
                    [0,0,1,0,0],
                    [0,1,1,1,0],
                    [0,0,1,0,0],
                    [0,0,0,0,0]])

kernel9 = np.array([[0,0,0,0,0],
                    [0,1,1,1,0],
                    [0,1,1,1,0],
                    [0,1,1,1,0],
                    [0,0,0,0,0]])

kernel21 = np.array([[0,1,1,1,0],
                     [1,1,1,1,1],
                     [1,1,1,1,1],
                     [1,1,1,1,1],
                     [0,1,1,1,0]])


def process(filename, calibrator, thresh=1, tmin=0, tmax=np.inf, xborder=0, verbose=False):

    header,px,py,highest,raw_region,timestamp,millistamp,images,dropped,millis_images = unpack_all(filename)

    cal_region = calibrator.calibrate_region(px,py,raw_region,header)
    icenter = raw_region.shape[1]//2
   
    threshold,prescale = get_trigger(header)

    tlim_cut = (millistamp > tmin) & (millistamp < tmax)
    thresh_cut = (highest==prescale.size) & (cal_region[:,icenter] >= thresh)
    border_cut = (px >= xborder) & (py < calibrator.width-xborder)
    keep = tlim_cut & thresh_cut & border_cut
 
    t = millistamp[keep]
    x = px[keep]
    y = py[keep]
    raw = raw_region[keep,icenter]
    cal = cal_region[keep,icenter]

    sum5 = (kernel5.flatten() * cal_region[keep]).sum(axis=1)
    sum9 = (kernel9.flatten() * cal_region[keep]).sum(axis=1)
    sum21 = (kernel21.flatten() * cal_region[keep]).sum(axis=1)

    t0 = millis_images[np.logical_not(np.isin(millis_images, t))]

    if verbose:
        print('Keeping', np.unique(t).size, '/', millis_images.size)


    return t, x, y, raw, cal, sum5, sum9, sum21, t0
 

def save_trig(fname, t, x, y, raw, cal, sum5, sum9, sum21, t0):

    f = r.TFile(args.out, 'recreate')
    trigs = r.TTree('triggers', 'Triggered events')

    sort = np.argsort(t)

    t_ = np.array([t[sort[0]]])
    max_ = np.array([cal[sort[0]]], dtype=np.uint32)
    sum5_ = np.array([sum5[sort[0]]], dtype=np.uint32)
    sum9_ = np.array([sum9[sort[0]]], dtype=np.uint32)
    sum21_ = np.array([sum21[sort[0]]], dtype=np.uint32)
    vx = r.vector('UInt_t')()
    vy = r.vector('UInt_t')()
    vraw = r.vector('UInt_t')()
    vcal = r.vector('UInt_t')() 

    trigs.Branch('t', t_, 't/l')
    trigs.Branch('max', max_, 'max/i')
    trigs.Branch('sum5', sum5_, 'sum5/i')
    trigs.Branch('sum9', sum9_, 'sum9/i')
    trigs.Branch('sum21', sum21_, 'sum21/i')
    trigs.Branch('x', vx)
    trigs.Branch('y', vy)
    trigs.Branch('raw', vraw)
    trigs.Branch('cal', vcal)
    
    for i in sort:
        if t[i] > t_[0]:
            trigs.Fill()

            t_[0] = t[i]
            max_[0] = cal[i]
            sum5_[0] = sum5[i]
            sum9_[0] = sum9[i]
            sum21_[0] = sum21[i]
            vx.clear()
            vy.clear()
            vraw.clear()
            vcal.clear()
            
        vx.push_back(int(x[i]))
        vy.push_back(int(y[i]))
        vraw.push_back(int(raw[i]))
        vcal.push_back(int(cal[i]))
        
        max_[0] = max(max_[0], cal[i])
        sum5_[0] = max(sum5_[0], sum5[i])
        sum9_[0] = max(sum9_[0], sum9[i])
        sum21_[0] = max(sum21_[0], sum21[i])
        
    trigs.Fill()

    nontrig = r.TTree('nontriggers', 'Empty frame timestamps')
    nontrig.Branch('t', t_, 't/l')
    
    for ti in np.sort(t0):
        t_[0] = ti
        nontrig.Fill()

    f.Write()
    f.Close()


if __name__ == "__main__":
    example_text = '''examples:

    ...'''
    
    parser = argparse.ArgumentParser(description='Select muon candidates from cosmic data.', epilog=example_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('trig', metavar='TRIG', nargs='+', help='trigger file(s) to process')
    parser.add_argument('--sandbox',action="store_true", help="run sandbox code and exit (for development).")
    parser.add_argument('--calib', default='calib', help="compare calibrated pixel values.")
    parser.add_argument('--tmin', type=float, default=0, help="minimum timestamp to save")
    parser.add_argument('--tmax', type=float, default=np.inf, help="maximum timestamp to save")
    parser.add_argument('--max',  type=int, default=50,help="maximum pixel value in rate plot (x-axis).")
    parser.add_argument('--plot', action='store_true',help='plot histogram of triggered values')
    parser.add_argument('--out', help='name of output ROOT file')
    parser.add_argument('--thresh', type=int, default=1, help='only save values at or above threshold')
    parser.add_argument('--no_hotcell', action='store_true', help='trigger without offline hotcell cleaning')
    parser.add_argument('--xborder', type=int, default=0, help='Cut x edge to change effective area')
    parser.add_argument('-v', '--verbose', action='store_true', help='enable verbose output')
    args = parser.parse_args() 

    end = "\n" if args.verbose else "\r"

    calibrator = Calibrator(args.calib, offline=(not args.no_hotcell))

    # update these lists
    all_t     = []  
    all_x     = []
    all_y     = []
    all_raw   = []
    all_cal   = []
    all_sum5  = []
    all_sum9  = []
    all_sum21 = []
    all_t0    = []

    n_images = 0

    for filename in args.trig:
        print("processing trigger file:  ", filename, end=end)
        t, x, y, raw, cal, sum5, sum9, sum21, t0 = process(filename, 
                calibrator,
                thresh=max(args.thresh, calibrator.blk_lvl+1), 
                tmin=args.tmin,
                tmax=args.tmax,
                xborder=args.xborder,
                verbose=args.verbose)

        all_t     += list(t)
        all_x     += list(x)
        all_y     += list(y)
        all_raw   += list(raw)
        all_cal   += list(cal)
        all_sum5  += list(sum5)
        all_sum9  += list(sum9)
        all_sum21 += list(sum21)
        all_t0    += list(t0)

    all_t     = np.array(all_t)
    all_x     = np.array(all_x)
    all_y     = np.array(all_y)
    all_raw   = np.array(all_raw)
    all_cal   = np.array(all_cal)
    all_sum5  = np.array(all_sum5)
    all_sum9  = np.array(all_sum9)
    all_sum21 = np.array(all_sum21)
    all_t0    = np.array(all_t0)

    time = np.unique(np.hstack([all_t, all_t0]))
    rate = time.size / (time.max() - time.min())
    
    
    print()

    print("frame rate       ", 1000*rate, "Hz")
    print("hits per frame:  ", all_t.size / time.size)
    print("minimum time:    ", np.min(time))
    print("maximum time:    ", np.max(time))
    print("start date:      ", datetime.fromtimestamp(np.min(time)/1000))
    print("end date:        ", datetime.fromtimestamp(np.max(time)/1000))

    if args.plot:
        plt.hist(all_cal, bins=np.arange(500), log=True)
        plt.show()

    if args.out:
        save_trig(args.out, 
                all_t, 
                all_x, 
                all_y, 
                all_raw, 
                all_cal, 
                all_sum5, 
                all_sum9, 
                all_sum21, 
                all_t0)
