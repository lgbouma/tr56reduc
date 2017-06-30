'''
do aperture photometry on OGLE TR 56b data.
'''

from __future__ import division, print_function

import numpy as np
from astropy.io import fits
from astropy.stats import mad_std, sigma_clipped_stats
from astropy.table import Table
from astropy.time import Time
from astropy import units as u, constants as c
from photutils import DAOStarFinder, aperture_photometry, CircularAperture, \
    IRAFStarFinder, CircularAnnulus
from photutils import make_source_mask
import os, pickle

datadir = '../data/'

# Toss out the first 4 frames -- they were for calibration.
fnames = [f for f in os.listdir(datadir) if f.startswith('tr56_')
        and int(f.split('_')[-1].split('.')[0]) >= 134 ]
fnames = np.sort(fnames)

bands = ['g', 'r', 'i', 'z']
ch_num = {'g':1, 'r':4, 'i':5, 'z':8}

# Get dicts of thresh, fwhm, radius, annuli_r, N comparison stars.
import define_arbitrary_parameters as arb

# Initial positions for TR56, to be sure starfinder algorithm catches it.
# careful: these are inverted values from ds9 image
r_0 = {
      'g': {'x_0':286*u.pix, 'y_0':256*u.pix},
      'r': {'x_0':387*u.pix, 'y_0':158*u.pix},
      'i': {'x_0':892*u.pix, 'y_0':139*u.pix},
      'z': {'x_0':625*u.pix, 'y_0':190*u.pix}
      }
# frame tr56_195.fits is right after had to turn off scope to slew, messing
# with the pointing. recovery requires new initial step
r_195 = {
      'g': {'x_0':286*u.pix, 'y_0':260*u.pix},
      'r': {'x_0':449*u.pix, 'y_0':172*u.pix},
      'i': {'x_0':831*u.pix, 'y_0':147*u.pix},
      'z': {'x_0':630*u.pix, 'y_0':204*u.pix},
      }

fluxs = {}

for band in ['g','z']:

    got_one = False

    ##################################
    # FIGURE OUT DIRECTORY STRUCTURE #
    ##################################
    dname = '{:s}_{:d}comp_radius{:d}_rin{:d}_rout{:d}_thresh{:d}'.format(
            band, arb.N_comp_stars[band], arb.radius[band],
            arb.annuli_r[band][0], arb.annuli_r[band][1], arb.thresh[band])
    writedir = '../results/'+dname+'/'
    phottabledir = '../results/'+dname+'/phot_table/'

    for d in [writedir, phottabledir]:
        if not os.path.exists(d):
            os.mkdir(d)
            if d == writedir:
                os.mkdir(writedir+'selected_comparison_stars')

    existing_phot_files = np.sort([f for f in os.listdir(phottabledir) if
                                   f.endswith('_'+band+'.pkl')])
    print('WRN! removing {:d} existing phot table pickles ({:s} band)'.
            format(len(existing_phot_files), band))
    for epf in existing_phot_files:
        os.remove(phottabledir+epf)

    # Outputs of make_master_bias_and_master_flat.py &
    # get_timestamp_in_BJD_TDB.py
    flat = pickle.load(open('../intermediates/master_flat_frame.pkl','rb'))
    bias = pickle.load(open('../intermediates/master_bias_frame.pkl','rb'))
    times_BJD_TDB = np.loadtxt('../intermediates/timestamps_BJD_TDB_converted.txt')

    fluxs[band] = []
    time_mask = []

    for index, fname in enumerate(fnames):

        hdulist = fits.open(datadir + fname)
        hdr = hdulist[0].header

        image = np.swapaxes(hdulist[ch_num[band]].data, 0, 1)
        # cast to float, else subtraction to negative values give nans.
        image = image.astype(np.float32)

        # We are ignoring dark current. (Assuming it is negligible). Then:
        # science_reduced = (science_raw - master bias) /
        #                        master flat_{reduced,normalized},
        # In constructing calibration frames:
        # Master bias frame is median of stack of biases. It will be subtracted
        #   directly from the science images.
        # Master flat frame is 
        #     median( (twilight image - master bias) / mean(twilight image) )

        # Modify the flat slightly: for the digital readout barrier, starfinder
        # algorithms get unhappy if you don't nan negative values. Verify the
        # cut is not serious with the middle assertion statement.
        flat_edge_cutoff = 0
        flat[band][flat[band] < flat_edge_cutoff] = np.nan
        assert np.nanmin(flat[band]) >= 5*flat_edge_cutoff

        image = (image - bias[band]) / flat[band]

        # ONE OPTION: Could make source mask to estimate std deviation after
        # applying calibration frames. Sigma clip out the sources (over 5
        # iterations).  The std deviation of the resulting image could be
        # applied for source identification threshold. OR just apply a hard
        # count threshold.
        # mask = make_source_mask(image, snr=2, npixels=7, dilate_size=11,
        #         sigclip_iters=5)
        # mean, median, std = sigma_clipped_stats(image, sigma=3, mask=mask)
        # print('Applied bias and flat frames. 5 sigma is {:.3g}'.format(5*std))

        print('Applied bias and flat frames.')

        # Source detection threshold empirically determined. 14px empirically
        # measured FWHM. Sources must be separated by at least 1 FWHM. sky=None
        # means that "starfind"
        # (http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?starfind) will estimate
        # the sky background. This affects only the output values of the object
        # peak, flux, and mag values.

        # First shot: IRAF star finder. Seems wonkier that DAO star finder.
        # starfind = IRAFStarFinder(fwhm=14,
        #                           threshold=thresh[band],
        #                           minsep_fwhm=0.7,
        #                           sky=None
        #                           )
        # Decided to use DAOStarFinder instead (it uses elliptical Gaussians).
        starfind = DAOStarFinder(fwhm=arb.fwhm[band],
                                 threshold=arb.thresh[band],
                                 sky=0.,
                                 exclude_border=True
                                 )
        sources = starfind(image)

        positions = (sources['xcentroid'], sources['ycentroid'])
        # JNW "the best choice of aperture radius is ~twice the FWHM of the
        # stellar image". But this field is crowded. Based on movies w/
        # overplotted apertures, smaller should be better.
        circ_apertures = CircularAperture(positions, r=arb.radius[band])
        annulus_apertures = CircularAnnulus(
                positions, r_in=arb.annuli_r[band][0], r_out=arb.annuli_r[band][1])
        apers = [circ_apertures, annulus_apertures]
        phot_table = aperture_photometry(image, apers)

        #################################
        # Local background subtraction. #
        #################################

        ## Method 1: Follow
        ## http://photutils.readthedocs.io/en/stable/photutils (...)
        ##       /aperture.html#local-background-subtraction
        bkg_mean = phot_table['aperture_sum_1'] / annulus_apertures.area()
        bkg_sum = bkg_mean * circ_apertures.area()
        final_sum = phot_table['aperture_sum_0'] - bkg_sum
        phot_table['residual_aperture_sum'] = final_sum

        ## Method 2: Follow
        ## https://github.com/astropy/photutils/pull/453, sigclipping away stars
        ## in the annulus.
        #ann_masks = annulus_apertures.to_mask(method='center')
        #ann_masked_data = [am.apply(image) for am in ann_masks]

        ## Sigma clip stars in annular aperture.
        #pre_sc_median = [np.nanmedian(amd[amd != 0])
        #                    for amd in ann_masked_data]

        #pre_sc_std = [
        #        (np.nanmedian(np.abs(amd[amd != 0] - pre_sc_median[ix])))*1.483
        #        for ix, amd in enumerate(ann_masked_data)
        #        ]

        #sigma_cut = 3
        #siginds = [
        #        ( (np.abs(amd[amd != 0] - pre_sc_median[ix])) <
        #        (sigma_cut * pre_sc_std[ix]) )
        #        for ix, amd in enumerate(ann_masked_data)
        #        ]

        #ann_masked_sigclipped_data = [
        #        amd[amd != 0][siginds[ix]]
        #        for ix,amd in enumerate(ann_masked_data)
        #        ]

        #bkg_median = np.array(
        #        [np.nanmedian(amds[amds != 0])
        #         for amds in ann_masked_sigclipped_data]
        #        )

        #bkg_sum = bkg_median*circ_apertures.area()

        #final_sum = phot_table['aperture_sum_0'] - bkg_sum
        #phot_table['residual_aperture_sum'] = final_sum

        #####################################
        # End local background subtraction. #
        #####################################

        if not got_one:
            # Initialize guesses handread from images until you find a match.
            x_0 = r_0[band]['x_0']
            y_0 = r_0[band]['y_0']
        elif fname == 'tr56_195.fits':
            # Use handread numbers post slewing issue.
            x_0 = r_195[band]['x_0']
            y_0 = r_195[band]['y_0']
        elif got_one:
            x_0 = previous_centroid[0]
            y_0 = previous_centroid[1]

        print('{:s}: {:s}band: {:d}/{:d}. x_0: {:.3g}, y_0: {:.3g}.'.format(
            fname, band, int(index), int(len(fnames)), x_0, y_0)+\
              ' Got {:d} sources'.format(len(phot_table))
            )

        dist = np.sqrt( (phot_table['xcenter']-x_0)**2 +
                        (phot_table['ycenter']-y_0)**2 )
        phot_table['dist_from_r0_px'] = dist
        phot_table.sort(['dist_from_r0_px'])

        pisco_pixelscale = u.pixel_scale(0.069*u.arcsec/u.pixel)
        phot_table['dist_from_r0_as'] = phot_table['dist_from_r0_px'].to(
                u.arcsec, pisco_pixelscale)

        tr56 = phot_table[0]

        if tr56['dist_from_r0_as'] > 1.*u.arcsec:
            # NOTE: this approach (if there is a jump, just skip to the
            # next frame and ignore the last one) was "needed" b/c of one frame
            # in i-band images: tr_45?.fits (see notes for the correct number).
            # If you don't do this, get stuck on that frame.

            print('ERR: {:s} position of tr56 is {:.3g} ({:.3g}) from prev'.\
            format(fname, tr56['dist_from_r0_px'], tr56['dist_from_r0_as']))
            print('\tcontinue to next frame...')

            time_mask.append(False)

            continue

        got_one = True

        assert tr56['dist_from_r0_as'] < 1.*u.arcsec,\
            '!ERR! {:s} position of tr56 is {:.3g} ({:.3g}) from guess'.\
            format(fname, tr56['dist_from_r0_px'], tr56['dist_from_r0_as'])

        time_mask.append(True)
        fluxs[band].append(float(tr56['residual_aperture_sum']))

        phot_table['time'] = times_BJD_TDB[index]

        pickle.dump(
                phot_table,
                open(phottabledir+'{:s}_{:s}.pkl'.
                    format(fname.split('.')[0], band), 'wb')
                )

        previous_centroid = (tr56['xcenter'], tr56['ycenter'])

    # NOTE: this is reset for every band, so you mask out different times
    # depending on where photometry is working.
    time_mask = np.array(time_mask, dtype=bool)
    t_BJD_TDB = times_BJD_TDB[time_mask]

    pickle.dump(
            t_BJD_TDB,
            open(writedir+'times_{:s}.pkl'.format(band), 'wb')
            )
    pickle.dump(
            np.array(fluxs[band]),
            open(writedir+'fluxs_{:s}.pkl'.format(band), 'wb')
            )
