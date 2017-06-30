from __future__ import division, print_function

import matplotlib.pyplot as plt
import pickle
import numpy as np

for band in ['r', 'i']:
    time = pickle.load(
           open('../tr56_results/times_{:s}.pkl'.format(band), 'rb')
           )

    flux = pickle.load(
           open('../tr56_results/fluxs_{:s}.pkl'.format(band), 'rb')
           )

    f, ax = plt.subplots()

    time = time.jd
    t_0 = int(np.floor(np.median(time)))

    thiscolor = 'black'

    ax.plot(time-t_0, flux, c=thiscolor, linestyle='-',
        marker='o', markerfacecolor=thiscolor,
        markeredgecolor=thiscolor, ms=1., lw=0.)

    ax.set_xlabel('JD(not calibrated) - {:d} [days]'.format(t_0))
    ax.set_ylabel('counts [adu]')

    f.tight_layout()

    f.savefig('../tr56_results/flux_vs_time_{:s}band.png'.format(band), dpi=300)
