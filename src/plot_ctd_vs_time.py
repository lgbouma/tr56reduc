from __future__ import division, print_function

import matplotlib.pyplot as plt
import pickle, os
import numpy as np

phottabledir = '../tr56_results/phot_table/'
resultsdir = '../tr56_results/'

for band in ['r', 'i']:

    time = pickle.load( open(resultsdir+'times_{:s}.pkl'.format(band), 'rb'))

    fs = np.sort([f for f in os.listdir(phottabledir) if
                  f.endswith('_'+band+'.pkl')])

    x_cen, y_cen = [], []
    for f in fs:

        phot_table = pickle.load(open(phottabledir+f, 'rb'))

        tr56 = phot_table[0]

        x_cen.append(tr56['xcenter'].value)
        y_cen.append(tr56['ycenter'].value)

    
    # NOTE: could convert timestamps to BJD_TDB (cf Eastman code)
    time = time.jd
    t_0 = int(np.floor(np.median(time)))
    x_cen, y_cen = np.array(x_cen), np.array(y_cen)

    thiscolor = 'black'

    # PLOT #1: true centroid vs time
    for index, cen in enumerate([x_cen, y_cen]):
        plt.close('all')
        f, ax = plt.subplots()

        ax.plot(time-t_0, cen, c=thiscolor, linestyle='-',
            marker='o', markerfacecolor=thiscolor,
            markeredgecolor=thiscolor, ms=1., lw=0.)

        ax.set_xlabel('JD(not calibrated) - {:d} [days]'.format(t_0))

        ystr = 'x_cen' if index==0 else 'y_cen'

        ax.set_ylabel(ystr+' [px]')

        f.tight_layout()

        f.savefig('../tr56_results/{:s}_vs_time_{:s}band.png'.
                format(ystr, band), dpi=300)

    # PLOT #2: shifted centroid vs time accounting for telescope repointing.
    for index, cen in enumerate([x_cen, y_cen]):

        plt.close('all')
        f, ax = plt.subplots()

        # NOTE: in the i-band, x_cen vs time there's a spike at the frame u
        # skipped. This messes up the diff. So choose the index from the 'r'
        # band (...)
        if band == 'r' and index == 0:
            r_ind = np.argmax(np.diff(cen))
            end_ind = r_ind
        if index == 1:
            end_ind = np.argmax(np.diff(cen))
        if band == 'i' and index == 1:
            end_ind = r_ind

        cen_pre  =  cen[:end_ind]
        t_pre  =  time[:end_ind]
        cen_post =  cen[end_ind+1:]
        t_post  =  time[end_ind+1:]

        thiscolor='blue'

        px_scale = 0.069 # arcsec/px
        ax.plot(t_pre-t_0, px_scale*(cen_pre-np.median(cen_pre)), c=thiscolor,
                linestyle='-', marker='o', markerfacecolor=thiscolor,
                markeredgecolor=thiscolor, ms=1., lw=0.)
        thiscolor='black'
        ax.plot(t_post-t_0, px_scale*(cen_post-np.median(cen_post)),
                c=thiscolor, linestyle='-', marker='o',
                markerfacecolor=thiscolor, markeredgecolor=thiscolor, ms=1.,
                lw=0.)

        ax.set_xlabel('JD(not calibrated) - {:d} [days]'.format(t_0))

        ystr = 'x_cen' if index==0 else 'y_cen'

        ax.set_ylabel(ystr+\
                ' [arcsec, arbitrary normlzn], blue: pre-repoint, black: post')

        f.tight_layout()

        f.savefig('../tr56_results/norm_{:s}_vs_time_{:s}band.png'.
                format(ystr, band), dpi=200)



