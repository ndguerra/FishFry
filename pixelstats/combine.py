#!/usr/bin/env python3
from   unpack import *
import numpy as np
import argparse
import sys
import re
import os

OUTFILE         = "combined.npz"
BASE_PATH       = "runs/"
PATH            = None

CHECK_EXPOSURE  = 0
CHECK_MIN       = 0
CHECK_MAX       = 0

PIXELS          = 0
SENS            = 0
EXPOSURE        = 0
WIDTH           = 0
HEIGHT          = 0
SECTION         = 0
SAMPLESTEP      = 0
FIRSTPIXEL      = 0
LASTPIXEL       = 0

SUM             = np.array([])
SSQ             = np.array([])
MAXES           = np.array([])
NUM             = np.array([])
SECOND          = np.array([])

def process(filename):
    global PIXELS, SENS, EXPOSURE, SUM, SSQ, MAXES, NUM, WIDTH, HEIGHT, \
            SECTION, SAMPLESTEP, FIRSTPIXEL, LASTPIXEL

    if (CHECK_MAX > 0):
        match = re.search(r'run_([\d]+)_', filename)
        run = int(match.group(1))
        if ((run < CHECK_MIN) or (run > CHECK_MAX)):
            return

    version, header, arr_sum, arr_ssq, arr_max, arr_second = unpack_all(filename)

    exposure     = interpret_header(header, "exposure")
    sens         = interpret_header(header, "sens")
    width        = interpret_header(header, "width")
    height       = interpret_header(header, "height")
    images       = interpret_header(header, "images")
    pixel_start  = interpret_header(header, "pixel_start")
    pixel_end    = interpret_header(header, "pixel_end")
    sample_step  = interpret_header(header, "sample_step")
    
    if CHECK_EXPOSURE > 0 and exposure != CHECK_EXPOSURE:
        #print "(skipping file ", filename, " with exposure ", exposure, ")"
        return

    print("processing file:  ", filename)


    # take parameters from first file:
    if PIXELS  == 0:
        PIXELS      = width*height
        SENS        = sens
        EXPOSURE    = exposure
        SUM         = np.zeros(PIXELS)
        SSQ         = np.zeros(PIXELS)
        NUM         = np.zeros(PIXELS)
        MAXES       = np.zeros([PIXELS,4])
        WIDTH       = width
        HEIGHT      = height
        SECTION     = pixel_start - pixel_end
        SAMPLESTEP  = sample_step
        FIRSTPIXEL  = pixel_start #fix here
        LASTPIXEL   = PIXELS      #fix here


    if exposure!=EXPOSURE or sens!=SENS:
        print("ERROR: inconsistent run parameters in file ", filename)
        unpack_header(filename, show=1)
        exit(0)

    SUM[pixel_start:pixel_end]    += arr_sum
    SSQ[pixel_start:pixel_end]    += arr_ssq
    NUM[pixel_start:pixel_end]    += images
    MAXES[pixel_start:pixel_end,1] = arr_max 
    MAXES[pixel_start:pixel_end,0] = arr_second 

    MAXES = np.sort(MAXES,1) 


def post():
    print("saving combined data as", PATH) 
    np.savez(PATH, 
            pixels=PIXELS, 
            sum=SUM, 
            ssq=SSQ, 
            max=MAXES[:,3], 
            second=MAXES[:,2],
            num=NUM, 
            exposure=EXPOSURE, 
            sens=SENS, 
            width=WIDTH, 
            height=HEIGHT, 
            section=SECTION, 
            samplestep=SAMPLESTEP, 
            firstpixel=FIRSTPIXEL, 
            lastpixel=LASTPIXEL)

if __name__ == "__main__":
    example_text = '''example:

    ./combine.py --exposure=50000 dark/*.dat --range 1242 1310 --out=dark_50000.npz'''

    parser = argparse.ArgumentParser(description='Combine multiple pixelstats data files.', epilog=example_text,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', metavar='FILE', nargs='+',
                    help='file to process')
    parser.add_argument('--exposure',type=int,help="use only runs with this exposure setting")
    parser.add_argument('--out',help="output filename")    
    parser.add_argument('--range', nargs=2, metavar=("FIRST","LAST"), type=int,help="only use runs where FIRST <= run number <= LAST")
    args = parser.parse_args()
    
    if args.range:
        CHECK_MIN = args.range[0]
        CHECK_MAX = args.range[1]
        print("using only runs with run number from ", CHECK_MIN, " to ", CHECK_MAX, " (inclusive).")

    if (args.exposure != None) and (args.exposure > 0):
        CHECK_EXPOSURE = args.exposure
        print("using only runs with exposure ", CHECK_EXPOSURE)

    if (args.out != None):
        PATH = args.out
    else:
        PATH = os.path.join(BASE_PATH, OUTFILE)

    for filename in args.files:
        process(filename)
    post()
        
