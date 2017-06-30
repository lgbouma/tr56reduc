'''
Make sure ur comparison stars looks reasonable w/ centroids, magnitudes,
colors, etc. Do this by overplotting the apertures on the images.

Also, plot flux vs time for all of them.
'''

from __future__ import division, print_function

import matplotlib.pyplot as plt
import pickle, os
import numpy as np

phottabledir = '../tr56_results/phot_table/'
datadir = '../data/170618_tr56_clay_PISCO/'
resultsdir = '../tr56_results/'
imgfnames = [f for f in os.listdir(datadir) if f.startswith('tr56_')
        and int(f.split('_')[-1].split('.')[0]) >= 134 ]
imgfnames = np.sort(datafnames)

for band in ['r', 'i']:

    time = pickle.load( open(resultsdir+'times_{:s}.pkl'.format(band), 'rb'))
    norm_tr56_flux = pickle.load(
            open(resultsdir+'norm_tr56_flux_{:s}.pkl'.format(band)))
    comp_d = pickle.load(
            open(resultsdir+'comparison_d_{:s}.pkl'.format(band)))

    #comp_d = {'flux_comp': flux_comp, 'cen_x_comp': cen_x_comp, 
    #          'cen_y_comp': cen_y_comp, 'comp_flux_signal': comp_flux_signal}
    
    # NOTE: could convert timestamps to BJD_TDB (cf Eastman code)
    time = time.jd
    t_0 = int(np.floor(np.median(time)))


    # get tr56 centroid positions
    cenfs = np.sort([f for f in os.listdir(phottabledir) if
                  f.endswith('_'+band+'.pkl')])
    x_cen, y_cen = [], []
    for f in cenfs:
        phot_table = pickle.load(open(phottabledir+f, 'rb'))
        tr56 = phot_table[0]
        x_cen.append(tr56['xcenter'].value)
        y_cen.append(tr56['ycenter'].value)
    x_cen, y_cen = np.array(x_cen), np.array(y_cen)

    # OK. Have times and centroid positions.
    # Now make images with overplotted centroid X's.
    # Do every fifth to save disk space.
    mult = 5
    for time_ind, fname in enumerate(datafnames[::mult]):
        print('{:s}-band: {:d}/{:d}, {:s}'.
              format(band, time_ind, len(datafnames[::mult]), fname))

        hdulist = fits.open(datadir + fname)
        hdr = hdulist[0].header

        image = np.swapaxes(hdulist[ch_num[band]].data, 0, 1)
        # cast to float, else subtraction to negative values give nans.
        image = image.astype(np.float32)

        # background subtraction (placeholder)
        image -= int(np.median(image))

        plt.close('all')
        f, ax  = plt.subplots()
        # NOTE: could tune vmin, vmax depending on ds9 params
        ax.imshow(np.log(image), vmin=np.log(800), vmax=np.log(64000),
                cmap=plt.get_cmap('gray_r'))

        thiscolor='red'
        ax.plot(x_cen[time_ind*mult], y_cen[time_ind*mult], c=thiscolor,
                linestyle='-', marker='x', markerfacecolor=thiscolor,
                markeredgecolor=thiscolor, ms=3., lw=1.)

        thiscolor='blue'
        # iterate over comparison stars
        for comp_ind in range(comp_d['cen_x_comp'].shape[1]):
            ax.plot(comp_d['cen_x_comp'][time_ind*mult,comp_ind],
                    comp_d['cen_y_comp'][time_ind*mult,comp_ind],
                    c=thiscolor,
                    linestyle='-',
                    marker='x',
                    markerfacecolor=thiscolor,
                    markeredgecolor=thiscolor,
                    ms=3.,
                    lw=1.)

        ax.set(xlim=(0,image.shape[1]), ylim=(image.shape[0],0))

        f.tight_layout()

        savename = band+'_'+fname.split('.')[0]+'.png'

        f.savefig(resultsdir+savename, dpi=150)

