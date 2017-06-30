import numpy as np
from astropy.io import fits
import os, pickle

def make_master_bias_and_master_flat():

    datadir = '../data/'

    # bias numbers 12 thru 20 are for the subraster that we're using.
    bias_fs = [f for f in os.listdir(datadir) if f.startswith('Bias_') and
               int(f.split('.')[0].split('_')[1]) >=12 ]

    ch_num = {'g':1, 'r':4, 'i':5, 'z':8}

    bias_stack_dict = {'g':[], 'r':[], 'i':[], 'z':[]}
    bias_median_dict = {'g':[], 'r':[], 'i':[], 'z':[]}
    N_flats = {'g':[], 'r':[], 'i':[], 'z':[]}

    # Create master bias frame
    bands = ['g','r','i','z']
    for band in bands:
        for ix, bias_f in enumerate(bias_fs):

            hdulist = fits.open(datadir+bias_f)

            hdr = hdulist[0].header

            image = np.swapaxes(hdulist[ch_num[band]].data, 0, 1)
            # cast to float, else subtraction to negative values give nans.
            image = image.astype(np.float32)

            bias_stack_dict[band].append(image)

            hdulist.close()

        bias_median_dict[band] = np.median(np.array(bias_stack_dict[band]), axis=0)

        N_flats[band].append(len(bias_stack_dict[band]))

    pickle.dump(bias_median_dict,
            open('../intermediates/master_bias_frame.pkl', 'wb'))

    # Create master flat frame.  "Sky flats" are images taken at twilight,
    # processed to remove the dark signal, normalized to unity, and then median
    # averaged to remove stars and reduce random noise.

    twiflat_fs = [f for f in os.listdir(datadir) if f.startswith('twiflat_') and
            int(f.split('.')[0].split('_')[1]) >=662 ]

    exp_time_dict = {'g':[], 'r':[], 'i':[], 'z':[]}
    flat_stack_dict = {'g':[], 'r':[], 'i':[], 'z':[]}
    flat_median_dict = {'g':[], 'r':[], 'i':[], 'z':[]}

    for band in bands:
        for ix, twiflat_f in enumerate(twiflat_fs):

            hdulist = fits.open(datadir+twiflat_f)

            hdr = hdulist[0].header

            image = np.swapaxes(hdulist[ch_num[band]].data, 0, 1)
            # cast to float, else subtraction to negative values give nans.
            image = image.astype(np.float32)

            # Remove "dark" signal (just bias, since we assume dark current can be
            # ignored).
            image -= bias_median_dict[band]

            # Normalize the image to unity. This accounts for variable exposure
            # time -- hdr['EXPTIME'] is not the same for each twilight flat.
            image /= np.mean(image)

            flat_stack_dict[band].append(image)

            hdulist.close()

        # Take the median of the normalized stack to remove stars, and reduce
        # random noise.
        flat_median_dict[band] = np.median(np.array(flat_stack_dict[band]), axis=0)

    pickle.dump(flat_median_dict,
            open('../intermediates/master_flat_frame.pkl', 'wb'))


def make_demo_reduced_frames():
    '''
    Did the flat fielding seem to work? I.e., does the sky level seem to be the
    same all over the image?  Did it get rid of artifacts and out-of-focus
    specks of dust, etc.?
    '''

    datadir = '../data/'
    # Toss out the first 4 frames -- they were for calibration.
    fnames = [f for f in os.listdir(datadir) if f.startswith('tr56_')
            and int(f.split('_')[-1].split('.')[0]) >= 134 ]
    fnames = np.sort(fnames)

    ch_num = {'g':1, 'r':4, 'i':5, 'z':8}
    bands = ['g','r','i','z']

    for band in bands:

        flat = pickle.load(open('../intermediates/master_flat_frame.pkl','rb'))
        bias = pickle.load(open('../intermediates/master_bias_frame.pkl','rb'))

        for index, fname in enumerate(fnames):

            if not ( (index % 20) == 0 ):
                continue

            hdulist = fits.open(datadir + fname)
            hdr = hdulist[0].header

            image = np.swapaxes(hdulist[ch_num[band]].data, 0, 1)
            image = image.astype(np.float32)

            flat_edge_cutoff = 0
            flat[band][flat[band] < flat_edge_cutoff] = np.nan
            assert np.nanmin(flat[band]) >= 5*flat_edge_cutoff

            image = (image - bias[band]) / flat[band]

            outdir = '../intermediates/reduced_frames/'
            outname = fname.split('_')[-1].split('.')[0] + '_' + band \
                        + '_reduced.fits'
            fits.writeto(outdir+outname, image, header=hdr)


if __name__ == '__main__':

    make_master_bias_and_master_flat()

    #make_demo_reduced_frames()
