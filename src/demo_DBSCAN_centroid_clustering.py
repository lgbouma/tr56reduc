import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from astropy.io import fits
import os, pickle

#########################
# Get the centroid data #
#########################

datadir = '../data/170618_tr56_clay_PISCO/'
phottabledir = '../tr56_results/phot_table/'
pkldir = '../tr56_results/'

N_other_stars = {'r':38, 'i':100}
N_comp_stars = {'r':5, 'i':5}
N_pre, N_post = 0, 0

for band in ['i']:

    time = pickle.load( open(pkldir+'times_{:s}.pkl'.format(band), 'rb'))

    phot_files = np.sort([f for f in os.listdir(phottabledir) if
                          f.endswith('_'+band+'.pkl')])

    # split by before/after zenith crossing.
    flux, cen_x, cen_y = [],[],[]
    flux_post, cen_x_post, cen_y_post = [],[],[]

    for f in phot_files:

        phot_table = pickle.load(open(phottabledir+f, 'rb'))

        if int(f.split('_')[1]) < 195:
            flux.append(np.array(phot_table['residual_aperture_sum']))
            cen_x.append(np.array(phot_table['xcenter']))
            cen_y.append(np.array(phot_table['ycenter']))
            N_pre +=1

        else:
            flux_post.append(np.array(phot_table['aperture_sum']))
            cen_x_post.append(np.array(phot_table['xcenter']))
            cen_y_post.append(np.array(phot_table['ycenter']))
            N_post +=1


    flux = np.concatenate(flux).ravel()
    cen_x = np.concatenate(cen_x).ravel()
    cen_y = np.concatenate(cen_y).ravel()

    flux_post = np.concatenate(flux_post).ravel()
    cen_x_post = np.concatenate(cen_x_post).ravel()
    cen_y_post = np.concatenate(cen_y_post).ravel()

###############################################################
# First, try doing the clustering on the pre-repointing data: #
# cen_x, cen_y                                                #
###############################################################

# X : array of shape [n_samples, n_features]
#     The data
# y : array of shape [n_samples]
#     The integer labels for cluster membership of each sample.

X = np.array([cen_x, cen_y]).T

# Standardize features by removing the mean and scaling to unit variance
# NOTE: this is not obviously OK.
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

# ~10 pixel drift means eps of 10.
# In pre-repointing data, ~4 pixel drift at most.
db = DBSCAN(eps=4, min_samples=N_pre/2).fit(X)

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))

###############
# Plot result #
###############

import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='none', markersize=3, alpha=0.2)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='none', markersize=2, alpha=0.2)

plt.title('Estimated number of clusters (stars): %d' % n_clusters_)

plt.savefig('../tr56_results/demo_centroid_clustering.png', dpi=250)
print('saved ../tr56_results/demo_centroid_clustering.png')

# OK, now add label text
for k, col in zip(unique_labels, colors):
    if k == -1:
        # skip
        continue

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    cen_x, cen_y = xy[:,0], xy[:,1]
    plt.text( np.nanmean(cen_x), np.nanmean(cen_y), k, fontsize=5 )

plt.savefig('../tr56_results/demo_centroid_clustering_with_labels.png', dpi=250)
print('saved ../tr56_results/demo_centroid_clustering_with_labels.png')
