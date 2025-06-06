#!/usr/bin/env python3

#
# gain.py:  combines multiple files under different exposure and light conditions to determine the best fit gain for every pixel
#

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse
import sys
import os
from unpack import *
from matplotlib.colors import LogNorm

# for stacking
SUMS       = []
SSQS       = []
MEANS      = []
VARIANCES  = []

# pixel information
total_pixels   = 0
first_pixel    = 0
last_pixel     = 0

WIDTH   = 0
HEIGHT  = 0 

def process(filename, args):
    global pixels, first_pixel, last_pixel, \
           SUMS, SSQS, MEANS, VARIANCES, \
           WIDTH, HEIGHT
    
    f = np.load(filename)

    if not WIDTH:
        total_pixels  = f['pixels']
        first_pixel   = f['firstpixel']
        last_pixel    = f['lastpixel']
        WIDTH         = f['width']
        HEIGHT        = f['height']

        print('total pixels: ', total_pixels)
    
    print('processing file: ', filename)

    # compute means and variances, append to global lists
    mean      = f['sum'] / f['num']
    variance  = (f['ssq'] / f['num'] - mean**2) * f['num'] / (f['num'] - 1)
    
    SUMS.append(f['sum'])
    SSQS.append(f['ssq'])
    MEANS.append(mean)
    VARIANCES.append(variance)

    f.close()
 
# Best-fit line, assuming constant fractional uncertainty in the y value.
# --> uses a weighted regression TBDetermined
def fit_line(x, y, xrange=(0,np.inf), yrange=(0,np.inf)):
    print('\nfitting line, divide by zero will occur')
    print('size of x: ', x.size)
    print('size of y: ', y.size)

    xmin, xmax = xrange
    ymin, ymax = yrange
   
    # weighted regression for calculating gain and intercept
    w = ((x >= xmin) & (x < xmax) & (y >= ymin) & (y < ymax)).astype(bool)

    xw = x*w
    yw = y*w

    sx     = np.sum(xw, axis=0)
    sy     = np.sum(yw, axis=0)
    sxx    = np.sum(xw**2, axis=0)
    sxy    = np.sum(xw*yw, axis=0)
    syy    = np.sum(yw**2, axis=0)
    sw     = np.sum(w, axis=0)

    # calculate R^2 values    
    r2 = (sxy*sw - sx*sy)**2 / (sxx*sw - sx**2) / (syy*sw - sy**2)

    a = (sxy*sw - sx*sy) / (sxx*sw - sx**2)
    b = (sy - a*sx)/sw
     
    print('line fitted. \nall intermediate values computed:')
    print('a:       ', a)
    print('b:       ', b)
    print('R^2:       ', r2)
    print('min: ', r2[np.logical_not(np.isnan(r2))].min())
    print('max: ', r2[np.logical_not(np.isnan(r2))].max())
    print()

    print(np.where(np.isnan(a)))
    
    return a, b, r2, sw


def plot_array(statistic, title, cmap='viridis', log=False):
    global WIDTH, HEIGHT

    sort_stat = np.sort(statistic)
    idx = statistic.size // 100

    vmin = sort_stat[idx]
    vmax = sort_stat[-idx]

    if log:
        norm = LogNorm(vmin=vmin, vmax=vmax)
        im = plt.imshow(statistic.reshape(HEIGHT, WIDTH), cmap=cmap, norm=norm, origin='lower')
    else:
        im = plt.imshow(statistic.reshape(HEIGHT, WIDTH), cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')

    plt.title(title)
    plt.xlabel('x [pixels]')
    plt.ylabel('y [pixels]')
    plt.colorbar()


def plot_points(ax, mean, variance, gain, intercept, rsq):
 
    idx = np.random.randint(gain.size)

    plt_x = mean[:,idx]
    plt_y = variance[:,idx]
    
    a = gain[idx]
    b = intercept[idx]
    x = np.linspace(0, 1.1*plt_x.max(), 100)
    y = a * x + b
    r2 = rsq[idx]

    ax.plot(plt_x, plt_y, 'o', linestyle='', )
    ax.plot(x, y, c='gold')
    ax.set_xlabel(r'$\mu$')
    ax.set_ylabel(r'$\sigma^2$')
    ax.set_title(r'$R^2={:.4f}$'.format(r2))  
        

def plot_hist(x, y, xlabel=None, ylabel=None, title=None):

    xs = np.sort(x)
    ys = np.sort(y)
    idx = xs.size // 5000
    
    xmin = xs[idx]
    ymin = ys[idx]
    xmax = xs[-idx]
    ymax = ys[-idx]
    '''
    if np.isnan(xmax):
        finite_x = np.delete(xs, np.where(np.isnan(xs)))
        xmax = np.max(finite_x)

    if np.isnan(ymax):
        finite_y = np.delete(ys, np.where(np.isnan(ys)))
        ymax = np.max(finite_y)

    print(np.where(np.isnan(y)))
    print(np.where(np.isnan(x)))
    print(len(x))
    '''
    _,_,_,im = plt.hist2d(x, y, bins=(200,200), 
            range=((xmin,xmax), (ymin, ymax)), 
            norm=LogNorm())
    
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)

    plt.colorbar(im, fraction=0.046, pad=0.04)



### old
def make_plots(args, gain, intercept, means, variances):
    print('making plots (in progress)')
    # find array of temporary max values, then find true maximums
    temp_xmax = np.max(means, axis=1)
    temp_ymax = np.max(variances,  axis=1)
    
    xmax = np.max(temp_xmax)
    ymax = np.max(temp_ymax)
    attributes('temp_xmax', temp_xmax)
    attributes('temp_ymax', temp_ymax)

    print('xmax: ', xmax)
    print('ymax: ', ymax)

    plt.close()
    f, axes = plt.subplots(3, 3, sharex='col', sharey='row')
    axes = axes.flatten()

    nonzero = means != 0
    attributes('nonzero', nonzero)
    plotted = 0
   
    '''
    Notes: 
    1) gets stuck after running through first plot
    2) perhaps just try to make one individual plot?
    3) plot means and variances of just one pixel?
    4) find the original s6 file that would work with Mike's previous gain script, 
    then individually print out values so you can figure it out from there
    '''

    for i in nonzero:
        mean = mean_stack[1][i]
        var  = var_stack[1][i]
        print('mean.shape: ', mean.shape)
        print('mean.size: ', mean.size) 
        print('var.shape: ', var.shape)
        print('var.size: ', var.size) 
        perm = np.argsort(mean)
        mean = mean[perm]
        var  = var[perm]

        a = gain[i]
        b = intercept[i]

        attributes('a', a)
        attributes('b', b)

        #fx = np.array([0, xmax])
        #attributes('fx', fx)
        #fy = a + b

        #axes[plotted].plot(fx,fy,"--k")

        axes[plotted].plot(mean,var,"ob")
        axes[plotted].set_xlim(0,xmax)        
        axes[plotted].set_ylim(0,ymax)    
        axes[plotted].xaxis.set_major_locator(ticker.LinearLocator(3))
        axes[plotted].xaxis.set_minor_locator(ticker.LinearLocator(xmax/20+1))
        axes[plotted].yaxis.set_major_locator(ticker.LinearLocator(3))
        axes[plotted].yaxis.set_minor_locator(ticker.LinearLocator(ymax/100+1))
        axes[plotted].text(0.3*xmax,0.8*ymax,"pixel "+str(first_pixel+i),horizontalalignment='center')

        plotted += 1
        print('plotted: ', plotted)
        if (plotted == 1):
            break
    axes[3].set_ylabel("variance")
    axes[7].set_xlabel("mean")
    plt.savefig("plots/gain.pdf")
    plt.show()

        
if __name__ == "__main__":
    example_text = '''examples:

    ./gain.py --calib ../calib --plot_all data/run_*.npz'''
    
    parser = argparse.ArgumentParser(description='Fit gain of each pixel from a series of runs at different exposures and light levels', epilog=example_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', metavar='FILE', nargs='+', help='file to process')
    parser.add_argument('--calib', default='calib', help="calibration directory to save files")
    parser.add_argument('--pixel_range',type=int,nargs=2, metavar=("MIN","MAX"),help="Only evalulate pixels with MIN <= index < MAX", default=[0,0])
    parser.add_argument('--mean_range',type=float,nargs=2, metavar=('XMIN', 'XMAX'), help="minimum and maximum mean to include in fitting", default=(0,850))
    
    parser.add_argument('--commit', action='store_true', help='commit to .npz files in calibration directory')
    parser.add_argument('-s', '--spatial_plots', action="store_true", help="make spatial plots for gain, black level, R^2, and number of sample points")
    parser.add_argument('-p', '--pix_plots', action='store_true', help='plot linear fits for 6 individual pixels')
    parser.add_argument('-S', '--scatterplot', action='store_true', help='plot histogram of R^2 vs gain for pixels')
    parser.add_argument('--plot_all', action='store_true', help='')
    parser.add_argument('--sandbox', action="store_true", help="experimental code")

    args = parser.parse_args()

    for filename in args.files:
        process(filename, args)
    
    # after completed processing, save and plot
    print('processing complete. \ntotal files processed: ', len(args.files)) 

    mean = np.vstack(MEANS)
    variance = np.vstack(VARIANCES)

    gain, intercept, rsq, count = fit_line(mean, variance, xrange=args.mean_range)
    black_level = -intercept / gain
    print('fitted gain complete.')

    n_figs = 0
    if args.spatial_plots or args.plot_all:
        n_figs += 1
        plt.figure(n_figs, figsize=(4.1, 2.1), tight_layout=True)
        plot_array(gain, title='Gain', cmap='plasma')
         
        n_figs += 1
        plt.figure(n_figs, figsize=(4.8, 2.6), tight_layout=True)
        plot_array(black_level, title='Black level', cmap='cool')
       
        n_figs += 1
        plt.figure(n_figs, figsize=(4.1, 2.1), tight_layout=True)
        plot_array(rsq, title='$R^2$', cmap='seismic') 

        n_figs += 1
        plt.figure(n_figs, figsize=(4.8, 2.6), tight_layout=True)
        plot_array(count, title='Data points', cmap='rainbow')
        

    if args.pix_plots or args.plot_all:
        n_figs += 1
        figsize = (7.5, 4.8)
        fig, axes = plt.subplots(2,3, figsize=figsize, tight_layout=True)

        for ax in axes.flatten():
            plot_points(ax, mean, variance, gain, intercept, rsq)
        
    if args.scatterplot or args.plot_all:
        n_figs += 1
        plt.figure(n_figs, figsize=(6,4))
        plot_hist(gain, rsq, 
                xlabel='Gain', ylabel=r'$R^2$')

    if n_figs:
        plt.show()

    if args.sandbox:
        print('========================================')
        print('-- in sandbox development environment -- ')
        print('========================================')
        print()

        attributes('gain', gain)
        attributes('intercept', intercept)
        attributes('R squared', rsq)
        
        fig, ax1 = plt.subplots()
        img1 = ax1.imshow(rsq.reshape(HEIGHT, WIDTH))
        plt.colorbar(img1, ax=ax1)
        
        fig, ax2 = plt.subplots()
        img2 = ax2.imshow(gain.reshape(HEIGHT, WIDTH))
        plt.colorbar(img2, ax=ax2)
        plt.show()

        print('plotting') 

    # commit computed information to .npz files
    if args.commit:
        print('saving fit line results')
        
        print('done. \ndumping information to be committed:')
        print('gain.shape:      ', gain.shape)
        print('gain.size:       ', gain.size)
        print('intercept.shape: ', intercept.shape)
        print('intercept.size:  ', intercept.size)
        print('count.shape:     ', count.shape)
        print('count.size:      ', count.size)
        print('means.shape:     ', mean.shape)
        print('means.size:      ', mean.size)
        print('vars.shape:      ', variance.shape)
        print('vars.size:       ', variance.size)
        
        np.savez(os.path.join(args.calib, "gain.npz"), 
                gain=gain, 
                intercept=intercept,
                rsq=rsq,
                count=count, 
                means=mean, 
                variances=variance)


