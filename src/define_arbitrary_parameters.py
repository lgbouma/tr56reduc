'''
Arbitrary parameters that matter a surprising amount for making photometry
reasonably good.
'''

__all__ = ['thresh', 'fwhm', 'radius', 'annuli_r']

# Thresholds for source identification after bias & flat fielding.
# These must be low enough to catch good comparison stars. Too high, and
# astrometry gets confused.
thresh = {'g':800, 'r':1700, 'i':2200, 'z':1100}

# FWHM for star finder. Set by coarsely measuring off ds9 images.
fwhm =   {'g':14, 'r':14, 'i':14, 'z':12}

# Radii for aperture photometry. Hand-measured.
radius = {'g':18, 'r':18, 'i':18, 'z':17}
annuli_r = {'g':[21,24], 'r':[21,24], 'i':[21,24], 'z':[20,23]}

# MANUAL CHOOSING: best for r: 8, i: (5 or 8, not sure)
N_comp_stars = {'r':8, 'i':5, 'g':6, 'z':6}
