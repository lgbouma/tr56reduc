import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os, pickle

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

import define_arbitrary_parameters as arb

eps_d = {'r':15, 'i':15}

datadir = '../data/'
phottabledir = '../results/phot_table/'

for band in ['r','i']:

    dname = '{:s}_{:d}comp_radius{:d}_rin{:d}_rout{:d}_thresh{:d}'.format(
            band, arb.N_comp_stars[band], arb.radius[band],
            arb.annuli_r[band][0], arb.annuli_r[band][1], arb.thresh[band])
    resultsdir = '../results/'+dname+'/'
    if not os.path.exists(resultsdir):
        os.mkdir(resultsdir)
        os.mkdir(resultsdir+'selected_comparison_stars')

    N_pre, N_post = 0, 0

    # get the centroid data (and photometry!)
    phot_files = np.sort([f for f in os.listdir(phottabledir) if
                          f.endswith('_'+band+'.pkl')])

    # split by before/after zenith crossing.
    flux, time, cen_x, cen_y = [],[],[],[]
    flux_post, time_post, cen_x_post, cen_y_post = [],[],[],[]

    for f in phot_files:

        phot_table = pickle.load(open(phottabledir+f, 'rb'))

        if int(f.split('_')[1]) < 195:
            flux.append(np.array(phot_table['residual_aperture_sum']))
            time.append(np.array(phot_table['time']))
            cen_x.append(np.array(phot_table['xcenter']))
            cen_y.append(np.array(phot_table['ycenter']))
            N_pre +=1

        else:
            flux_post.append(np.array(phot_table['residual_aperture_sum']))
            time_post.append(np.array(phot_table['time']))
            cen_x_post.append(np.array(phot_table['xcenter']))
            cen_y_post.append(np.array(phot_table['ycenter']))
            N_post +=1

    flux = np.concatenate(flux).ravel()
    time = np.concatenate(time).ravel()
    cen_x = np.concatenate(cen_x).ravel()
    cen_y = np.concatenate(cen_y).ravel()

    flux_post = np.concatenate(flux_post).ravel()
    time_post = np.concatenate(time_post).ravel()
    cen_x_post = np.concatenate(cen_x_post).ravel()
    cen_y_post = np.concatenate(cen_y_post).ravel()

    ###############################################################
    # Do the clustering on the pre-repointing data, and then the  #
    # post-repointing data                                        #
    ###############################################################

    # X : array of shape [n_samples, n_features]
    #     The data to cluster on. For us, 2 features are centroid_x and centroid_y. 
    #     We have of order 1,000,000 "samples" (centroid values for every star for
    #     every time).
    # y : array of shape [n_samples]
    #     The integer labels for cluster membership of each sample.
    # Z : array of shape [n_samples, n_features]
    #     The extra data (not clustered on). The two features are flux and
    #     time. This is to allow later pairing btwn cluster labels and
    #     flux+time (along w/ centroids).

    for ix, (c_x, c_y, f, t) in enumerate([(cen_x, cen_y, flux, time),
            (cen_x_post, cen_y_post, flux_post, time_post)]):

        if ix == 0:
            print('beginning pre-repointing clustering')
        elif ix == 1:
            print('beginning post-repointing clustering')

        X = np.array([c_x, c_y]).T
        Z = np.array([f, t]).T

        # Standard ML practice is to standardize features by removing the mean and
        # scaling to unit variance This is not obviously OK in our case, so we skip
        # this step.

        # X = StandardScaler().fit_transform(X)

        # DBSCAN:
        # Perform DBSCAN clustering from vector array or distance matrix.
        # DBSCAN - Density-Based Spatial Clustering of Applications with Noise.
        # Finds core samples of high density and expands clusters from them.
        # Good for data which contains clusters of similar density.
        # Good comparison stars must have centroid info at least half of the time.
        # eps : float, optional
        #     The maximum distance between two samples for them to be considered
        #     as in the same neighborhood.
        # 
        # min_samples : int, optional
        #     The number of samples (or total weight) in a neighborhood for a point
        #     to be considered as a core point. This includes the point itself.

        if ix == 0:
            eps = 4  # maybe ~ 4 pixels drift in the pre-pointing data.
            min_samples = 0.8*N_pre
        if ix == 1:
            eps = eps_d[band] # much more drift in the post-repointing data
            min_samples = 0.8*N_post

        print('doing DBSCAN. {:d} features, {:d} clusters'.
              format(X.shape[0], X.shape[1]))
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)

        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        print('Estimated number of clusters: %d' % n_clusters_)

        # The silhouette coefficient is a nice way to see how good ur data matches
        # something "clusterable". However it's expensive to compute, so don't.
        # print("Silhouette Coefficient: %0.3f"
        #       % metrics.silhouette_score(X, labels))

        outdict = {'X':X, 'Z':Z, 'labels':labels,
                'core_samples_mask':core_samples_mask}

        substr = 'pre-repointing_'+band if ix==0 else 'post-repointing_'+band
        pickle.dump( outdict,
            open(resultsdir+'centroid_cluster_{:s}.pkl'.format(substr),'wb'))

        ##########################
        # Plot result: no labels #
        ##########################
        plt.close('all')
        # Black removed and is used for noise instead.
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(unique_labels))]

        for k, col in zip(unique_labels, colors):
            mec, alpha = 'none', 0.2
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]
                mec = 'black'
                alpha = 1

            class_member_mask = (labels == k)

            xy = X[class_member_mask & core_samples_mask]
            cen_x, cen_y = xy[:,0], xy[:,1]
            plt.plot(cen_x, cen_y, 'o', markerfacecolor=tuple(col),
                     markeredgecolor=mec, markersize=3, alpha=alpha)

            xy = X[class_member_mask & ~core_samples_mask]
            plt.plot(cen_x, cen_y, 'o', markerfacecolor=tuple(col),
                     markeredgecolor=mec, markersize=2, alpha=alpha)

        plt.title('Estimated number of clusters (stars): %d' % n_clusters_)

        plt.savefig(resultsdir+'centroid_clustering_{:s}.png'.format(substr), dpi=250)
        print('saved {:s}centroid_clustering_{:s}.png'.format(resultsdir,substr))

        ############################
        # Plot result: with labels #
        ############################
        for k, col in zip(unique_labels, colors):
            if k == -1:
                continue
            class_member_mask = (labels == k)

            xy = X[class_member_mask & core_samples_mask]
            cen_x, cen_y = xy[:,0], xy[:,1]
            plt.text( np.nanmean(cen_x), np.nanmean(cen_y), k, fontsize=5 )

        plt.savefig(resultsdir+'centroid_clustering_{:s}_with_labels.png'.
                format(substr), dpi=250)
        print('saved {:s}centroid_clustering_{:s}_with_labels.png'.
                format(resultsdir,substr))


        ###########################################
        # Plot result: with labels, color by flux #
        ###########################################
        import matplotlib.cm as cmx
        import matplotlib.colors as colors
        plt.close('all')

        unique_labels = set(labels)
        median_fluxs = []

        # Get median fluxes for colormap.
        for k in unique_labels:
            if k == -1:
                continue
            class_member_mask = (labels == k)
            ft = Z[class_member_mask]
            flux, time = ft[:,0], ft[:,1]
            median_fluxs.append(np.nanmedian(flux))

        # Now plot, colored by median flux
        log_median_fluxs = np.array(np.log10(median_fluxs))
        viridis = plt.cm.get_cmap('viridis')
        c_norm = colors.Normalize(vmin=np.nanmin(log_median_fluxs),
                                  vmax=np.nanmax(log_median_fluxs))
        scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=viridis)

        for k, lmf in zip(unique_labels, log_median_fluxs):
            if k == -1:
                continue

            class_member_mask = (labels == k)

            color_val = scalar_map.to_rgba(lmf)

            xy = X[class_member_mask & core_samples_mask]
            cen_x, cen_y = xy[:,0], xy[:,1]
            plt.plot(cen_x, cen_y, 'o', markerfacecolor=color_val,
                     markeredgecolor='none', markersize=3, alpha=alpha)

            xy = X[class_member_mask & ~core_samples_mask]
            plt.plot(cen_x, cen_y, 'o', markerfacecolor=color_val,
                     markeredgecolor='none', markersize=2, alpha=alpha)

        for k in unique_labels:
            if k == -1:
                continue
            class_member_mask = (labels == k)

            xy = X[class_member_mask & core_samples_mask]
            cen_x, cen_y = xy[:,0], xy[:,1]
            plt.text( np.nanmean(cen_x), np.nanmean(cen_y), k, fontsize=5 )

        plt.title('Estimated number of clusters (stars): %d' % n_clusters_)

        plt.savefig(resultsdir+'centroid_clustering_fluxcolor_{:s}.png'.format(substr), dpi=250)
        print('saved {:s}centroid_clustering_fluxcolor_{:s}.png'.
                format(resultsdir, substr))

