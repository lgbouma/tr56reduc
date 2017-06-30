import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os, pickle

datadir = '../data/170618_tr56_clay_PISCO/'
phottabledir = '../tr56_results/phot_table/'
pkldir = '../tr56_results/'

N_other_stars = {'r':38, 'i':100}
N_comp_stars = {'r':5, 'i':5}

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

        else:
            flux_post.append(np.array(phot_table['residual_aperture_sum']))
            cen_x_post.append(np.array(phot_table['xcenter']))
            cen_y_post.append(np.array(phot_table['ycenter']))


    flux = np.concatenate(flux).ravel()
    cen_x = np.concatenate(cen_x).ravel()
    cen_y = np.concatenate(cen_y).ravel()

    flux_post = np.concatenate(flux_post).ravel()
    cen_x_post = np.concatenate(cen_x_post).ravel()
    cen_y_post = np.concatenate(cen_y_post).ravel()


f, ax = plt.subplots()
ax.set(xlabel='x cen', ylabel='y cen')
plt.scatter(cen_x, cen_y, s=2, alpha=0.2, c='red')
plt.scatter(cen_x_post, cen_y_post, s=2, alpha=0.05, c='blue')
plt.title('red: before repointing. blue: after')
plt.tight_layout()
plt.savefig('../tr56_results/demo_project_centroids_on_time.png',dpi=250)
plt.close('all')

f, ax = plt.subplots()
ax.set(xlabel='x cen', ylabel='y cen')
plt.scatter(cen_x, cen_y, s=2, alpha=0.2, c='red')
plt.title('red: before repointing.')
plt.tight_layout()
plt.savefig('../tr56_results/demo_project_centroids_on_time_before_repointing.png',dpi=250)

f, ax = plt.subplots()
ax.set(xlabel='x cen', ylabel='y cen')
plt.scatter(cen_x_post, cen_y_post, s=2, alpha=0.05, c='blue')
plt.title('blue: after repointing')
plt.tight_layout()
plt.savefig('../tr56_results/demo_project_centroids_on_time_after_repointing.png',dpi=250)

