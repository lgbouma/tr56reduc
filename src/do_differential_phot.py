'''
You have:
1) tables of star positions & fluxes (at every given time)

2) approximate labels for stars obtained by running a clustering algorithm on
the time-projected centroid values.

Select comparison stars.
Get a comparison flux signal.
Also record comparison centroid signals.
'''

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clip
from astrobase.lcmath import time_bin_magseries
import os, pickle
from scipy.interpolate import interp1d

# Get dicts of thresh, fwhm, radius, annuli_r
import define_arbitrary_parameters as arb

datadir = '../data/'

for band in ['r','i']:

    dname = '{:s}_{:d}comp_radius{:d}_rin{:d}_rout{:d}_thresh{:d}'.format(
            band, arb.N_comp_stars[band], arb.radius[band],
            arb.annuli_r[band][0], arb.annuli_r[band][1], arb.thresh[band])
    phottabledir = '../results/'+dname+'/phot_table/'
    resultsdir = '../results/'+dname+'/'

    if not os.path.exists(resultsdir):
        os.mkdir(resultsdir)
        os.mkdir(resultsdir+'selected_comparison_stars')

    print('beginning {:s} band'.format(band))

    time_tr56 = pickle.load( open(resultsdir+'times_{:s}.pkl'.format(band), 'rb'))

    phot_files = np.sort([f for f in os.listdir(phottabledir) if
                          f.endswith('_'+band+'.pkl')])

    cen_x_tr56, cen_y_tr56, flux_tr56 = [], [], []

    for f in phot_files:

        phot_table = pickle.load(open(phottabledir+f, 'rb'))

        tr56 = phot_table[0]
        other_stars = phot_table[1:]

        flux_tr56.append(np.array(tr56['residual_aperture_sum']))
        cen_x_tr56.append(np.array(tr56['xcenter']))
        cen_y_tr56.append(np.array(tr56['ycenter']))

    assert len(time_tr56) == len(flux_tr56)

    # We want to select comparison stars. We have already clustered on
    # time-projected centroids to get labels for groupings of centroids that
    # are thought to be "the same star".

    cd = pickle.load(
            open(resultsdir+'centroid_cluster_post-repointing_{:s}.pkl'.
                format(band),'rb'))
    X, Z, labels = cd['X'], cd['Z'], cd['labels']
    core_samples_mask = cd['core_samples_mask']

    # Comparison stars must have reasonably (but not strictly) continuous
    # centroids (just like TR 56). This translates to meaning "the cluster must
    # have between 0.8*N_observations and 1*N_observations features" (or else
    # there are too few continuous centroid points, or too many [a likely
    # blend]). NOTE N_observations currently means after repointing.

    lower_frac, upper_frac = 0.7, 1.05

    unique_labels = set(labels)

    comp_dict = {}
    N_blend, N_interperr = 0, 0
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

        time_comp = time
        flux_comp = flux

        # FIXME: from here on, perform reduction for beyond repointing time
        # only. (tr56_195.fits, time '2017-06-19 05:06:36.000', or
        # 2457923.7129166666 JD, which is 2457923.719510681 BJD_TDB from Jason
        # Eastman's calculator)
        tr56_time_mask = (time_tr56 >= 2457923.7195106)
        flux_tr56 = np.vstack(flux_tr56).flatten()
        flux_tr56 = flux_tr56[tr56_time_mask]
        time_tr56 = time_tr56[tr56_time_mask]

        if len(time_comp) < lower_frac*len(time_tr56) or \
           len(time_comp) > upper_frac*len(time_tr56):
            # Skip stars that are blends or do not have enough centroid data.
            N_blend += 1
            continue
        if np.all(np.array_equal(flux, flux_tr56)):
            # Skip TR 56 (do not compare it to itself).
            continue

        # Interpolate to deal with gaps in measured fluxes for comparison
        # stars.
        fn = interp1d(time_comp, flux_comp, kind='linear')

        try:
            flux_comp_interp = fn(time_tr56)
        except ValueError:
            # Skip stars that do not have times at the beginning or end of TR
            # 56's observation times.
            N_interperr += 1
            continue

        # Store the info for candidate comparison stars.
        comp_dict[k] = {'c_x':cen_x, 'c_y':cen_y, 'f':flux_comp,
                        'f_interp':flux_comp_interp, 't':time_comp,
                        'chisq': np.sum((flux_tr56 - flux_comp_interp)**2 ) }


    print('cand comparison stars: {:d}/{:d}. {:d} interp err, {:d} blend'.
          format(
          len(list(comp_dict.keys())),
          len(unique_labels),
          N_interperr,
          N_blend )
         )

    N_comp_star = arb.N_comp_stars[band]

    # i: 32 noisy.
    veto_list = {
            'r':[],
            'i':[32]}

    ##Method 1: choose comparison stars based on chi squared. This winds up
    ##being oddly biased for reasons I don't understand.
    #label_chisq = [(label, comp_dict[label]['chisq']) for label in comp_dict]
    #selected_comp_labels = [label for (label, csq) in sorted(label_chisq,
    #                        key=lambda lc: lc[1]) if label != 0 and label not
    #                        in veto_list[band]][:N_comp_star]

    # Method 2: Choose comparison stars based on if they're close & similar
    # flux. (Manual selection, following centroid_clustering output).
    # NOTE: these comparison labels are actually only good for
    if dname=='r_8comp_radius18_rin21_rout24_thresh1700':
        selected_comp_labels = [57,72,23,39,30,28,40,4,17]
    elif dname=='r_8comp_radius15_rin20_rout24_thresh1700':
        selected_comp_labels = [57,72,23,39,30,28,40,4,17]
    elif dname=='i_5comp_radius18_rin21_rout24_thresh2200':
        selected_comp_labels = [10,59,42,115,98,63,110,4]
    elif dname=='i_5comp_radius15_rin20_rout24_thresh2200':
        selected_comp_labels = [10,59,42,115,98,63,110,4]
    elif dname=='g_6comp_radius18_rin21_rout24_thresh600':
        selected_comp_labels = [25,20,32,20,42,29,32,10,11]
    elif dname=='z_6comp_radius17_rin20_rout23_thresh1100':
        selected_comp_labels = [5,40,17,15,27,22,90,51,96]

    good_comp_stars = np.intersect1d(selected_comp_labels,
            list(comp_dict.keys()))
    print('N good comp stars: {:d}'.format(len(good_comp_stars)))

    selected_comp_labels = good_comp_stars[:arb.N_comp_stars[band]]

    assert len(selected_comp_labels) > 0
    assert len(selected_comp_labels) == arb.N_comp_stars[band]
    print('N selected comp stars: {:d}'.format(len(selected_comp_labels)))

    ############################################
    # Compute the comparison star flux signal. #
    ############################################
    # f_target,diffentialphot = f_target * weight,
    # where weight is defined as
    # Method 1: (sum over comparison stars of comp star fluxes)^{-1}
    # Method 2: (sum over comparison stars of [comp star flux / median of comp
    #                   star flux])^{-1}
    # Method 3: (sum over comparison stars of [comp star flux /
    #                   sqrt(median of comp star flux)])^{-1}
    # The weight is arbitrary, so whatever minimizes RMS is chosen as "tha
    # best".

    ix, flux_comp_list = 0, []

    for cl in selected_comp_labels:

        time_comp = comp_dict[cl]['t']
        flux_comp = comp_dict[cl]['f']
        flux_comp_interp = comp_dict[cl]['f_interp']

        # Sigma clip outliers from comparison star fluxes.
        median_flux = np.median(flux_comp_interp)
        stddev_flux = (np.median(np.abs(flux_comp_interp - median_flux))) * 1.483
        sigma_cut = 3
        sigind = (np.abs(flux_comp_interp - median_flux)) < \
                 (sigma_cut * stddev_flux)
        sflux_comp_interp = flux_comp_interp[sigind]
        stime_comp_interp = time_tr56[sigind]

        # Reinterpolate over sigma clipped points (FML).
        fn = interp1d(stime_comp_interp, sflux_comp_interp, kind='linear')
        sflux_comp_reinterp = fn(time_tr56)

        if ix == 0:
            print('WRN: removing old comparison star plots')
            compstarfs = [f for f in
                    os.listdir(resultsdir+'selected_comparison_stars/') if
                    f.endswith('_{:s}band.png'.format(band))]
            for csf in compstarfs:
                os.remove(resultsdir+'selected_comparison_stars/'+csf)

        ############################################################
        # Plot TR 56 flux vs time and comparison star flux vs time #
        ############################################################
        plt.close('all')
        f,axs= plt.subplots(nrows=2,ncols=1,figsize=(8,4))

        t_0_tr56 = int(np.floor(np.median(time_tr56)))
        axs[0].plot(time_tr56 - t_0_tr56, flux_tr56, c='black', linestyle='-',
                marker='o', markerfacecolor='black', markeredgecolor='black',
                ms=1, lw=0)
        axs[0].set(xlim=[min(time_tr56 - t_0_tr56),max(time_tr56 - t_0_tr56)])

        axs[1].plot(time_tr56 - t_0_tr56, sflux_comp_reinterp, c='black',
                linestyle='-', marker='o', markerfacecolor='black',
                markeredgecolor='black', ms=1, lw=0)
        axs[1].set(xlim=[min(time_tr56 - t_0_tr56),max(time_tr56 - t_0_tr56)])

        axs[1].set_xlabel('BJD TDB - {:d} [days]'.format(t_0_tr56))
        for a in [axs[0],axs[1]]:
            a.set_ylabel('counts [adu]')

        f.suptitle(str(cl), y=0.98)

        f.tight_layout()

        f.savefig(resultsdir+'selected_comparison_stars/'+\
                '{:s}_{:s}band.png'.format(str(cl), band),
                dpi=200, bbox_inches='tight')

        # Add star to comparison star flux list (stack created after).
        flux_comp_list.append(sflux_comp_reinterp)
        ix += 1

    # Stack comparison fluxes using different methods.
    flux_comp_arr = np.array(flux_comp_list)
    flux_comp_stack = {
            'method1': np.sum(flux_comp_arr, axis=0),
            'method2': np.sum(
                        flux_comp_arr/np.median(flux_comp_arr, axis=0),
                        axis=0),
            'method3': np.sum(
                        flux_comp_arr/np.sqrt(np.median(flux_comp_arr,axis=0)),
                        axis=0)
            }


    for method in np.sort(list(flux_comp_stack.keys())):

        ###########################
        # Plot TR 56 flux vs time #
        ###########################
        plt.close('all')
        f,ax = plt.subplots(figsize=(8,4))

        norm_flux_tr56 = flux_tr56/flux_comp_stack[method]
        norm_flux_tr56 /= np.median(norm_flux_tr56)

        # Sigma clip outliers from TR 56 fluxes.
        median_flux = np.median(norm_flux_tr56)
        stddev_flux = (np.median(np.abs(norm_flux_tr56- median_flux))) * 1.483
        sigma_cut = 3
        sigind = (np.abs(norm_flux_tr56- median_flux)) < \
                 (sigma_cut * stddev_flux)
        snorm_flux_tr56 = norm_flux_tr56[sigind]
        stime_tr56 = time_tr56[sigind]

        ax.plot(stime_tr56 - t_0_tr56, snorm_flux_tr56 ,
                c='black', linestyle='-', marker='o', markerfacecolor='black',
                markeredgecolor='black', ms=1, lw=0)

        bd = time_bin_magseries(stime_tr56, snorm_flux_tr56,
                binsize=240., minbinelems=5)
        bin_flux_tr56, bin_time_tr56 = bd['binnedmags'], bd['binnedtimes']
        ax.plot(bin_time_tr56 - t_0_tr56, bin_flux_tr56 ,
                c='black', linestyle='-', marker='o', markerfacecolor='red',
                markeredgecolor='none', ms=4, lw=2, alpha=0.5)

        rms = np.sqrt( np.sum( (snorm_flux_tr56 - np.median(snorm_flux_tr56) )**2 ) / \
                    len(snorm_flux_tr56) )
        ax.text(0.02, 0.02, 'rms: {:.3g}'.format(rms),
                verticalalignment='bottom', horizontalalignment='left',
                fontsize='small', transform=ax.transAxes,
                bbox=dict(facecolor='white', edgecolor='white', pad=1))

        ax.set_xlim([min(time_tr56 - t_0_tr56),max(time_tr56 - t_0_tr56)])
        ax.set_xlabel('BJD TDB - {:d} [days]'.format(t_0_tr56))
        ax.set_ylabel('relative flux ({:s})'.format(method))

        f.tight_layout()
        f.savefig(resultsdir+'tr56_flux_vs_time_{:s}band_{:s}.png'.
                format(band,method), dpi=300, bbox_inches='tight')

        #############################################
        # Plot:                                     #
        # sum of flux in adu of all the comp stars, #
        # flux of TR-56 in adu,                     #
        # TR-56 divided by comparison signal.       #
        #############################################
        plt.close('all')
        f,axs = plt.subplots(nrows=3, ncols=1, figsize=(8,8))

        # flux of TR-56 in adu
        axs[0].plot(stime_tr56 - t_0_tr56, flux_tr56[sigind],
                c='black', linestyle='-', marker='o', markerfacecolor='black',
                markeredgecolor='black', ms=1, lw=0)
        axs[0].set_ylabel('tr56 signal [ADU]', fontsize='x-small')

        # sum of flux in adu of all the comp stars
        axs[1].plot(stime_tr56 - t_0_tr56, flux_comp_stack[method][sigind],
                c='black', linestyle='-', marker='o', markerfacecolor='black',
                markeredgecolor='black', ms=1, lw=0)
        axs[1].set_ylabel('comparison signal ({:s}) [ADU]'.format(method),
                fontsize='x-small')

        # TR-56 divided by comparison signal.
        axs[2].plot(stime_tr56 - t_0_tr56, snorm_flux_tr56 ,
                c='black', linestyle='-', marker='o', markerfacecolor='black',
                markeredgecolor='black', ms=1, lw=0)

        bd = time_bin_magseries(stime_tr56, snorm_flux_tr56,
                binsize=240., minbinelems=5)
        bin_flux_tr56, bin_time_tr56 = bd['binnedmags'], bd['binnedtimes']
        axs[2].plot(bin_time_tr56 - t_0_tr56, bin_flux_tr56 ,
                c='black', linestyle='-', marker='o', markerfacecolor='red',
                markeredgecolor='none', ms=4, lw=2, alpha=0.5)
        axs[2].set_xlabel('BJD TDB - {:d} [days]'.format(t_0_tr56))
        axs[2].set_ylabel('(tr56/comparison) / median(tr56)',
                fontsize='x-small')

        for ax in axs:
            ax.set_xlim([min(time_tr56 - t_0_tr56),max(time_tr56 - t_0_tr56)])
            ax.set_xticklabels([])

        f.tight_layout()
        f.savefig(resultsdir+'tr56_compsignal_flux_vs_time_{:s}band_{:s}.png'.
                format(band, method), dpi=300, bbox_inches='tight')
