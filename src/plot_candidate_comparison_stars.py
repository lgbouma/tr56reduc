import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.io import fits
import os, pickle

datadir = '../data/170618_tr56_clay_PISCO/'
phottabledir = '../tr56_results/phot_table/'
pkldir = '../tr56_results/'

N_other_stars = {'r':38, 'i':100}
N_comp_stars = {'r':5, 'i':5}

for band in ['r']:

    time_tr56 = pickle.load( open(pkldir+'times_{:s}.pkl'.format(band), 'rb'))
    time_tr56 = time_tr56.jd

    phot_files = np.sort([f for f in os.listdir(phottabledir) if
                          f.endswith('_'+band+'.pkl')])

    # flux_comp: a list (of length number of time observations) where each
    # entry is the astropy table of the fluxes of the comparison stars.
    # cen_x_comp, cen_y_comp: the same, for the centroid values.
    cen_x_tr56, cen_y_tr56, flux_tr56 = [], [], []

    for f in phot_files:

        phot_table = pickle.load(open(phottabledir+f, 'rb'))

        tr56 = phot_table[0]
        other_stars = phot_table[1:]

        flux_tr56.append(np.array(tr56['residual_aperture_sum']))
        cen_x_tr56.append(np.array(tr56['xcenter']))
        cen_y_tr56.append(np.array(tr56['ycenter']))

        # NOTE: can remove
        # flux_other.append(np.array(other_stars['residual_aperture_sum']))
        # cen_x_other.append(np.array(other_stars['xcenter']))
        # cen_y_other.append(np.array(other_stars['ycenter']))

    assert len(time_tr56) == len(flux_tr56)

    # NOTE: flux_other, cen_*_other are now lists of arrays. The arrays have
    # different lengths -- depending on how many stars DAOStarFind got.
    # Each array is ordered by its distance from TR 56.

    # We want to select comparison stars. We have already clustered on
    # time-projected centroids to get labels for groupings of centroids that
    # are thought to be "the same star".

    cd = pickle.load(
            open('../tr56_intermediates/centroid_cluster_post-repointing_{:s}.pkl'.
                format(band),'rb'))
    X, Z, labels = cd['X'], cd['Z'], cd['labels']
    core_samples_mask = cd['core_samples_mask']

    # Comparison stars must have reasonably (but not strictly) continuous
    # centroids (just like TR 56). This translates to meaning "the cluster must
    # have between 0.9*N_observations and 1*N_observations features" (or else
    # there are too few continuous centroid points, or too many [a likely
    # blend]).

    lower_frac, upper_frac = 0.9, 1

    unique_labels = set(labels)

    comp_dict = {}
    for k in unique_labels:
        if k == -1:
            # Skip things classified as noise.
            continue

        class_member_mask = (labels == k)

        # Take both the core samples and the neighborhood (non-core) samples
        # into account when selecting comparison stars.
        xy = X[class_member_mask]
        cen_x, cen_y = xy[:,0], xy[:,1]

        ft = Z[class_member_mask]
        flux, time = ft[:,0], ft[:,1]

        if len(xy) < lower_frac*len(time) or len(xy) > upper_frac*len(time):
            # Skip stars that are blends or do not have enough centroid data.
            continue
        if np.all(flux == flux_tr56):
            # Skip TR 56, which cannot be compared to itself.
            continue

        # comp_dict stores the info for what we think might be OK comparison
        # stars.
        #comp_dict[k] = {'c_x':cen_x, 'c_y':cen_y, 'f':flux, 't':time,
        #        'f_sum':np.sum(flux), 'chisq':np.sum( (flux_tr56 - flux)**2 )}

        comp_dict[k] = {'c_x':cen_x, 'c_y':cen_y, 'f':flux, 't':time,
                'f_sum':np.nan, 'chisq':np.nan}

    for label in np.sort(list(comp_dict.keys())):

        plt.close('all')
        f,ax = plt.subplots(nrows=2,ncols=1,figsize=(6,3))
        
        t_0_tr56 = int(np.floor(np.median(time_tr56)))
        ax[0].plot(time_tr56 - t_0_tr56, flux_tr56, c='black', linestyle='-',
                marker='o', markerfacecolor='black', markeredgecolor='black',
                ms=1, lw=0)
        ax[0].set(xlim=[min(time_tr56 - t_0_tr56),max(time_tr56 - t_0_tr56)])

        cand_time = Time(comp_dict[label]['t'], format='iso', scale='utc')
        cand_time = cand_time.jd

        ax[1].plot(cand_time - t_0_tr56, comp_dict[label]['f'], c='black',
                linestyle='-', marker='o', markerfacecolor='black',
                markeredgecolor='black', ms=1, lw=0)

        ax[1].set(xlim=[min(time_tr56 - t_0_tr56),max(time_tr56 - t_0_tr56)])

        ax[1].set_xlabel('JD(not calibrated) - {:d} [days]'.format(t_0_tr56))
        for a in [ax[0],ax[1]]:
            a.set_ylabel('counts [adu]')

        f.suptitle(str(label), y=1.02)

        f.tight_layout()
        f.savefig('../tr56_results/candidate_comparison_stars/{:s}_{:s}.png'.
                format(str(label), band), dpi=200)


