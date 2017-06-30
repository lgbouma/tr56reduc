'''
Make sure ur comparison stars looks reasonable w/ centroids, magnitudes,
colors, etc. Do this by overplotting the apertures on the images.

Also, plot flux vs time for all of them.
'''

from __future__ import division, print_function

import matplotlib.pyplot as plt
import pickle, os
from astropy.io import fits
import numpy as np

phottabledir = '../tr56_results/phot_table/'
datadir = '../data/170618_tr56_clay_PISCO/'
resultsdir = '../tr56_results/'
writedir = '../tr56_results/imgs_comparison_centroid_overplot/'
imgfnames = [f for f in os.listdir(datadir) if f.startswith('tr56_')
        and int(f.split('_')[-1].split('.')[0]) >= 134 ]
imgfnames = np.sort(imgfnames)

ch_num = {'g':1, 'r':4, 'i':5, 'z':8}

flat = pickle.load(open('../tr56_intermediates/master_flat_frame.pkl','rb'))
bias = pickle.load(open('../tr56_intermediates/master_bias_frame.pkl','rb'))

for band in ['r', 'i']:

    time = pickle.load( open(resultsdir+'times_{:s}.pkl'.format(band), 'rb'))
    norm_tr56_flux = pickle.load(
            open(resultsdir+'norm_tr56_flux_{:s}.pkl'.format(band), 'rb'))

    cd = pickle.load(
            open('../tr56_intermediates/centroid_cluster_post-repointing_{:s}.pkl'.
                format(band),'rb'))
    X, Z, labels = cd['X'], cd['Z'], cd['labels']
    core_samples_mask = cd['core_samples_mask']

    cen_x, cen_y = X[:,0], X[:,1]
    #FIXME: this approach is dated (but not really necessary).


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
    for time_ind, fname in enumerate(imgfnames[::mult]):
        print('{:s}-band: {:d}/{:d}, {:s}'.
              format(band, time_ind, len(imgfnames[::mult]), fname))

        hdulist = fits.open(datadir + fname)
        hdr = hdulist[0].header

        image = np.swapaxes(hdulist[ch_num[band]].data, 0, 1)
        image = image.astype(np.float32)
        flat_edge_cutoff = 0
        flat[band][flat[band] < flat_edge_cutoff] = np.nan
        assert np.nanmin(flat[band]) >= 5*flat_edge_cutoff

        image = (image - bias[band]) / flat[band]

        plt.close('all')
        f, ax  = plt.subplots()
        # NOTE: could tune vmin, vmax depending on ds9 params
        ax.imshow(np.log(image), vmin=np.log(800), vmax=np.log(48000),
                cmap=plt.get_cmap('gray_r'))

        thiscolor='red'
        ax.plot(x_cen[time_ind*mult], y_cen[time_ind*mult], marker='o',
                c=thiscolor, markersize=25, markeredgewidth=1,
                markeredgecolor=thiscolor, markerfacecolor='None')

        thiscolor='blue'
        # iterate over comparison stars
        for comp_ind in range(len(cen_x)):

            ax.plot(cen_x[time_ind*mult,comp_ind],
                    cen_x][time_ind*mult,comp_ind],
                    marker='o',
                    c=thiscolor,
                    markersize=25,
                    markeredgewidth=1,
                    markerfacecolor='None',
                    markeredgecolor=thiscolor)


        ax.minorticks_on()
        ax.grid(True, which='both')
        ax.set(xlim=(0,image.shape[1]), ylim=(image.shape[0],0))

        f.tight_layout()

        savename = band+'_'+fname.split('.')[0]+'.png'

        f.savefig(writedir+savename, dpi=150)
