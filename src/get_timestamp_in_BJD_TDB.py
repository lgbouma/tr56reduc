'''
Use measured calibration between TELUT time and UTC (as it was listed on
the local clocks, presumably being updated by NIST). Once we have JD_UTC,
convert to BJD_TDB with
http://astroutils.astronomy.ohio-state.edu/time/utc2bjd.html.

N.b.: this needs to be run before do_simple_aperture_phot.py
'''

from __future__ import division, print_function

from astropy.io import fits
from astropy.table import Table
from astropy import units as u, constants as c
from astropy.time import Time
import numpy as np
import os, pickle

# Toss out the first 4 frames -- they were for calibration.
datadir = '../data/'
fnames = [f for f in os.listdir(datadir) if f.startswith('tr56_')
        and int(f.split('_')[-1].split('.')[0]) >= 134 ]
fnames = np.sort(fnames)

bands = ['g', 'r', 'i', 'z']
ch_num = {'g':1, 'r':4, 'i':5, 'z':8}

times = {}
exptimes = {}

for band_ix, band in enumerate(bands):

    times[band] = []
    exptimes[band] = []

    for index, fname in enumerate(fnames):

        hdulist = fits.open(datadir + fname)
        hdr = hdulist[0].header

        exptimes[band].append(hdr['EXPTIME'])
        times[band].append(hdr['DATEOBS']+' '+hdr['TELUT'])

    TELUT_t = Time(times[band], format='iso', scale='utc')

    midexposure_t_UTC = TELUT_t.jd*u.day - 4*u.second \
                            - 0.5*np.array(exptimes[band])*u.second
    midexposure_t_UTC = midexposure_t_UTC.value

    if band_ix > 0:
        assert np.all(np.array_equal(_, midexposure_t_UTC)), \
                    'Timestamps should be the same in every band.'

    _ = midexposure_t_UTC

# Since they're the same, just dump a single one.
np.savetxt(
        open( '../intermediates/timestamps_JD_UTC_to_convert.txt','wb'),
        midexposure_t_UTC,
        fmt='%.8f')
