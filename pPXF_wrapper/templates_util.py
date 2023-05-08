###############################################################################
#
# Copyright (C) 2016, Michele Cappellari
# E-mail: michele.cappellari_at_physics.ox.ac.uk
#
# This software is provided as is without any warranty whatsoever.
# Permission to use, for non-commercial purposes is granted.
# Permission to modify for personal or internal use is granted,
# provided this copyright and disclaimer are included unchanged
# at the beginning of the file. All other rights are reserved.
#
###############################################################################

# This file contains the 'miles' class with functions to contruct a
# library of MILES templates and interpret and display the output
# of pPXF when using those templates as input.


from __future__ import print_function

from os import path
import glob

import numpy as np
from scipy import ndimage
from astropy.io import fits
import matplotlib.pyplot as plt

import ppxf.ppxf_util as util
import ppxf.ppxf as ppxf
import sys
import astropy.units as u

def readcol(filename, **kwargs):
    """
    Tries to reproduce the simplicity of the IDL procedure READCOL.
    Given a file with some columns of strings and columns of numbers, this
    function extract the columns from a file and places them in Numpy vectors
    with the proper type:

    name, mass = readcol('prova.txt', usecols=(0, 2))

    where the file prova.txt contains the following:

    ##################
    # name radius mass
    ##################
      abc   25.   36.
      cde   45.   56.
      rdh   55    57.
      qtr   75.   46.
      hdt   47.   56.
    ##################

    This function is a wrapper for numpy.genfromtxt() and accepts the same input.
    See the following website for the full documentation
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.genfromtxt.html

    """
    f = np.genfromtxt(filename, dtype=None, **kwargs)

    t = type(f[0])
    if t == np.ndarray or t == np.void:  # array or structured array
        f = map(np.array, zip(*f))

    # In Python 3.x all strings (e.g. name='NGC1023') are Unicode strings by defauls.
    # However genfromtxt() returns byte strings b'NGC1023' for non-numeric columns.
    # To have the same behaviour in Python 3 as in Python 2, I convert the Numpy
    # byte string 'S' type into Unicode strings, which behaves like normal strings.
    # With this change I can read the string a='NGC1023' from a text file and the
    # test a == 'NGC1023' will give True as expected.

    if sys.version >= '3':
        f = [v.astype(str) if v.dtype.char == 'S' else v for v in f]

    return f

###############################################################################
# MODIFICATION HISTORY:
#   V1.0.0: Written. Michele Cappellari, Oxford, 27 November 2016


def age_metal(filename):
    """
    Extract the age and metallicity from the name of a file of
    the MILES library of Single Stellar Population models as
    downloaded from http://miles.iac.es/ as of 2016

    :param filename: string possibly including full path
        (e.g. 'miles_library/Mun1.30Zm0.40T03.9811.fits')
    :return: age (Gyr), [M/H]

    """
    # Mbi1.30Zm0.25T00.0300_iTp0.00_baseFe_linear_FWHM_2.51.fits
    # THIS IS NEW ADDED FOR OTHER FILENAMES FROM THE MILES WEBSITE INCLUDING FWHM INFORMATION
    s = path.basename(filename)
    s = s.split('_')[0]
#    print(s)
    age = float(s[s.find("T")+1:s.find(".fits")])
    metal = s[s.find("Z")+1:s.find("T")]
    if "m" in metal:
        metal = -float(metal[1:])
    elif "p" in metal:
        metal = float(metal[1:])
    else:
        raise ValueError("This is not a standard MILES filename")

    return age, metal

###############################################################################
# MODIFICATION HISTORY:
#   V1.0.0: Adapted from my procedure setup_spectral_library() in
#       ppxf_example_population_sdss(), to make it a stand-alone procedure.
#     - Read the characteristics of the spectra directly from the file names
#       without the need for the user to edit the procedure when changing the
#       set of models. Michele Cappellari, Oxford, 28 November 2016
#   V1.0.1: Check for files existence. MC, Oxford, 31 March 2017


def get_FWHM_EMILES(lam):
    # E-miles in SINFONI range: sig = 60 km/s
    # 1 / R = Δλ / λ = v / c
    sig = 60
    c = 299792458/1e3  # in km/s
    delta_lam = sig/c * lam*2.355  # lam in microns #NOPE not working??
    #delta_lam = 0.00046705998033431625*lam

    return delta_lam  # of E-MILES templates


def get_FWHM_SINFONI(lam):
    R = 4000
    delta_lam = lam/R
    return delta_lam


def get_FWHM_IRTF(lam):
    R = 2000
    delta_lam = lam/R
    return delta_lam


def get_FWHM_MUSE(lam):
    # from Gueron 2017
    #for MUSE
    fwhm = 6.266*1e-8 * lam**2 - 9.824*1e-4 * lam + 6.286
    return fwhm #in AA


def get_FWHM_diff(lam, lam_lim=10000, fwhm=-1):
    FWHM_diff = np.zeros_like(lam)
    mask = lam < lam_lim
    if fwhm == -1:
        fwhm_MUSE = get_FWHM_MUSE(lam[mask])
        mask_low = fwhm_MUSE < 2.51
        fwhm_MUSE[mask_low] = 2.51
        FWHM_diff[mask] = np.sqrt(fwhm_MUSE**2 - 2.51**2)  # gueron for MUSE part
    else:
        FWHM_diff[mask] = np.sqrt(fwhm**2 - 2.51**2)

    mask2 = lam >= lam_lim
    FWHM_diff[mask2] = 0
    return FWHM_diff


class miles(object):

    def __init__(self, pathname, velscale, fwhm=-1, normalize=False, age_lim=None, metal_lim=None,
                 instrument=None, wavelength_unit=u.AA, ssp_model_label='EMILES'):
        """
        Produces an array of logarithmically-binned templates by reading
        the spectra from the Single Stellar Population (SSP) library by
        Vazdekis et al. (2010, MNRAS, 404, 1639) http://miles.iac.es/.
        The code checks that the model specctra form a rectangular grid
        in age and metallicity and properly sorts them in both parameters.
        The code also returns the age and metallicity of each template
        by reading these parameters directly from the file names.
        The templates are broadened by a Gaussian with dispersion
        sigma_diff = np.sqrt(sigma_gal**2 - sigma_tem**2).

        Thie script relies on the files naming convention adopted by
        the MILES library, where SSP spectra have the form below

            Mun1.30Zm0.40T03.9811.fits

        This code can be easily adapted by the users to deal with other stellar
        libraries, different IMFs or different abundances.

        :param pathname: path with wildcards returning the list files to use
            (e.g. 'miles_models/Mun1.30*.fits'). The files must form a Cartesian grid
            in age and metallicity and the procedure returns an error if they are not.
        :param velscale: desired velocity scale for the output templates library in km/s
            (e.g. 60). This is generally the same or an integer fraction of the velscale
            of the galaxy spectrum.
        :param FWHM_gal: vector or scalar of the FWHM of the instrumental resolution of
            the galaxy spectrum in Angstrom.
        :param normalize: set to True to normalize each template to mean=1.
            This is useful to compute light-weighted stellar population quantities.
        :return: The following variables are stored as attributes of the miles class:
            .templates: array has dimensions templates[npixels, n_ages, n_metals];
            .log_lam_temp: natural np.log() wavelength of every pixel npixels;
            .age_grid: (Gyr) has dimensions age_grid[n_ages, n_metals];
            .metal_grid: [M/H] has dimensions metal_grid[n_ages, n_metals].
            .n_ages: number of different ages
            .n_metal: number of different metallicities
        """
        files = glob.glob(pathname)
        assert len(files) > 0, "Files not found %s" % pathname

        all = [age_metal(f) for f in files]
        all_ages, all_metals = np.array(all).T
        ages, metals = np.unique(all_ages), np.unique(all_metals)
        #print(ages, metals)

        assert set(all) == set([(a, b) for a in ages for b in metals]), \
            'Ages and Metals do not form a Cartesian grid'

        # Extract the wavelength range and logarithmically rebin one spectrum
        # to the same velocity scale of the SDSS galaxy spectrum, to determine
        # the size needed for the array which will contain the template spectra.
        hdu = fits.open(files[0])
        ssp = hdu[0].data
        h2 = hdu[0].header
        lam_range_temp = h2['CRVAL1'] + np.array([0, h2['CDELT1']*(h2['NAXIS1']-1)]) #in AA

        ssp_wave = np.linspace(lam_range_temp[0], lam_range_temp[-1], len(ssp))
        
        if instrument == 'MUSE':
            mask = (ssp_wave > 4300) & (ssp_wave < 9800)
        
        if instrument == 'SINFONI':
            wavelength_unit = u.micron
            ssp_wave = ssp_wave / 1e4 #to micron
            mask = (ssp_wave > 1.9) & (ssp_wave < 2.6)
        
        ssp_wave = ssp_wave[mask]
        ssp = ssp[mask]

        sspNew, log_lam_temp = util.log_rebin(
            [ssp_wave[0], ssp_wave[-1]], ssp, velscale=velscale)[:2]

        # Choose templates with age limit!
        if not age_lim is None:
            if np.isscalar(age_lim):
                ages = ages[ages >= age_lim]
            elif len(age_lim) == 2:
                mask_age = (ages >= age_lim[0]) & (ages < age_lim[1])
                ages = ages[mask_age]
            else:
                print('Wrong input for age_lim!')
        if not metal_lim is None:
            if np.isscalar(metal_lim):
                metals = metals[metals >= metal_lim]
            elif len(metal_lim) == 2:
                mask_metals = (metals >= metal_lim[0]) & (metals < metal_lim[1])
                metals = metals[mask_metals]
            else:
                print('Wrong input for age_lim!')

        n_ages = len(ages)
        n_metal = len(metals)

        templates = np.empty((sspNew.size, n_ages, n_metal))
        templates_lin = np.empty((ssp_wave.size, n_ages, n_metal))
        age_grid = np.empty((n_ages, n_metal))
        metal_grid = np.empty((n_ages, n_metal))

        # Convolve the whole Vazdekis library of spectral templates
        # with the quadratic difference between the galaxy and the
        # Vazdekis instrumental resolution. Logarithmically rebin
        # and store each template as a column in the array TEMPLATES.

        # Quadratic sigma difference in pixels Vazdekis --> galaxy
        # The formula below is rigorously valid if the shapes of the
        # instrumental spectral profiles are well approximated by Gaussians.

        # FWHM_dif = np.sqrt(get_FWHM_SINFONI(ssp_wave)**2 - get_FWHM_EMILES(ssp_wave)**2)
        # sigma = FWHM_dif/2.355/h2['CDELT1']
        # print(FWHM_dif)

        FWHM_dif = get_FWHM_diff(ssp_wave, fwhm=fwhm)

        sigma = FWHM_dif/2.355/h2['CDELT1']  # sigma in template pixels

        # Here we make sure the spectra are sorted in both [M/H] and Age
        # along the two axes of the rectangular grid of templates.

        for j, age in enumerate(ages):
            for k, metal in enumerate(metals):
                p = all.index((age, metal))
                hdu = fits.open(files[p])
                ssp = hdu[0].data
                ssp = ssp[mask]
                if np.isscalar(sigma):
                    ssp = ndimage.gaussian_filter1d(ssp, sigma)
                else:
                    ssp = util.gaussian_filter1d(ssp, sigma)  # convolution with variable sigma
                sspNew = util.log_rebin([ssp_wave[0], ssp_wave[-1]], ssp, velscale=velscale)[0]

                if normalize:
                    sspNew /= np.mean(sspNew)  # changes to luminosity weighted
                # K. FAH: ADDED THIS LINE FOR THE EMILES TEMPLATES
                sspNew[np.isnan(sspNew)] = 0

                templates_lin[:, j, k] = ssp
                templates[:, j, k] = sspNew
                age_grid[j, k] = age
                metal_grid[j, k] = metal

        self.templates = templates/np.median(templates)  # Normalize by a scalar, mass weighted SSPs
        self.templates_lin = templates_lin/np.nanmedian(templates_lin)
        self.wave_linear = ssp_wave
        self.log_lam_temp = log_lam_temp
        self.ssp_model_label = ssp_model_label
        self.age_grid = age_grid
        self.metal_grid = metal_grid
        self.n_ages = n_ages
        self.n_metal = n_metal
        self.velscale = velscale
        self.wavelength_unit = wavelength_unit
        


###############################################################################
# MODIFICATION HISTORY:
#   V1.0.0: Written. Michele Cappellari, Oxford, 1 December 2016
#   V1.0.1: Use path.realpath() to deal with symbolic links.
#       Thanks to Sam Vaughan (Oxford) for reporting problems.
#       MC, Garching, 11/JAN/2016


    def mass_to_light(self, weights, band="r", quiet=False):
        """
        Computes the M/L in a chosen band, given the weights produced
        in output by pPXF. A Salpeter IMF is assumed (slope=1.3).

        This procedure uses the photometric predictions
        from Vazdekis+12 and Ricciardelli+12
        http://adsabs.harvard.edu/abs/2012MNRAS.424..157V
        http://adsabs.harvard.edu/abs/2012MNRAS.424..172R
        they were downloaded in December 2016 below and are included in pPXF with permission
        http://www.iac.es/proyecto/miles/pages/photometric-predictions/based-on-miuscat-seds.php

        :param weights: pPXF output with dimensions weights[miles.n_ages, miles.n_metal]
        :param band: possible choices are "U", "B", "V", "R", "I", "J", "H", "K" for
            the Vega photometric system and "u", "g", "r", "i" for the SDSS AB system.
        :param quiet: set to True to suppress the printed output.
        :return: mass_to_light in the given band
        """
        assert self.age_grid.shape == self.metal_grid.shape == weights.shape, \
            "Input weight dimensions do not match"

        vega_bands = ["U", "B", "V", "R", "I", "J", "H", "K"]
        sdss_bands = ["u", "g", "r", "i"]
        vega_sun_mag = [5.600, 5.441, 4.820, 4.459, 4.148, 3.711, 3.392, 3.334]
        sdss_sun_mag = [6.55, 5.12, 4.68, 4.57]  # values provided by Elena Ricciardelli

        file_dir = path.dirname(path.realpath(__file__))  # path of this procedure

        if band in vega_bands:
            k = vega_bands.index(band)
            sun_mag = vega_sun_mag[k]
            file2 = file_dir + "/miles_models/Vazdekis2012_ssp_phot_Padova00_UN_v10.0.txt"
        elif band in sdss_bands:
            k = sdss_bands.index(band)
            sun_mag = sdss_sun_mag[k]
            file2 = file_dir + "/miles_models/Vazdekis2012_ssp_sdss_miuscat_UN1.30_v9.txt"
        else:
            raise ValueError("Unsupported photometric band")

        file1 = file_dir + "/miles_models/Vazdekis2012_ssp_mass_Padova00_UN_baseFe_v10.0.txt"
        slope1, MH1, Age1, m_no_gas = readcol(file1, usecols=[1, 2, 3, 5])

        slope2, MH2, Age2, mag = readcol(file2, usecols=[1, 2, 3, 4 + k])

        # The following loop is a brute force but very safe and general
        # way of matching the photometric quantities to the SSP spectra.
        # It makes no assumption on the sorting and dimensions of the files
        mass_no_gas_grid = np.empty_like(weights)
        lum_grid = np.empty_like(weights)
        for j in range(self.n_ages):
            for k in range(self.n_metal):
                p1 = (np.abs(self.age_grid[j, k] - Age1) < 0.001) & \
                     (np.abs(self.metal_grid[j, k] - MH1) < 0.01) & \
                     (np.abs(1.30 - slope1) < 0.01)
                mass_no_gas_grid[j, k] = m_no_gas[p1]

                p2 = (np.abs(self.age_grid[j, k] - Age2) < 0.001) & \
                     (np.abs(self.metal_grid[j, k] - MH2) < 0.01) & \
                     (np.abs(1.30 - slope2) < 0.01)
                lum_grid[j, k] = 10**(-0.4*(mag[p2] - sun_mag))

        # This is eq.(2) in Cappellari+13
        # http://adsabs.harvard.edu/abs/2013MNRAS.432.1862C
        mlpop = np.sum(weights*mass_no_gas_grid)/np.sum(weights*lum_grid)

        if not quiet:
            print('M/L_' + band + ': %.4g' % mlpop)

        return mlpop


###############################################################################


    def plot(self, weights, nodots=False, colorbar=True, **kwargs):

        assert self.age_grid.shape == self.metal_grid.shape == weights.shape, \
            "Input weight dimensions do not match"

        xgrid = np.log10(self.age_grid) + 9
        ygrid = self.metal_grid
        util.plot_weights_2d(xgrid, ygrid, weights,
                             nodots=nodots, colorbar=colorbar, **kwargs)


##############################################################################


    def mean_age_metal(self, weights, quiet=False):

        assert self.age_grid.shape == self.metal_grid.shape == weights.shape, \
            "Input weight dimensions do not match"

        log_age_grid = np.log10(self.age_grid) + 9
        metal_grid = self.metal_grid

        # These are eq.(1) and (2) in McDermid+15
        # http://adsabs.harvard.edu/abs/2015MNRAS.448.3484M
        mean_log_age = np.sum(weights*log_age_grid)/np.sum(weights)
        mean_metal = np.sum(weights*metal_grid)/np.sum(weights)

        if not quiet:
            print('Weighted <logAge> [yr]: %.3g' % mean_log_age)
            print('Weighted <[M/H]>: %.3g' % mean_metal)

        return mean_log_age, mean_metal


##############################################################################
class miles_grid(object):

    def __init__(self, pathname, age_lim=None, instrument=None):
        """
        Same as miles, but only to get the age and metallicity grid, no real templates.
        This is used for plotting purposes
        """
        files = glob.glob(pathname)
        assert len(files) > 0, "Files not found %s" % pathname

        all = [age_metal(f) for f in files]
        all_ages, all_metals = np.array(all).T
        ages, metals = np.unique(all_ages), np.unique(all_metals)

        assert set(all) == set([(a, b) for a in ages for b in metals]), \
            'Ages and Metals do not form a Cartesian grid'

        # Choose templates of fixed age!

        if not age_lim is None:
            ages = ages[ages >= age_lim]

        n_ages = len(ages)
        n_metal = len(metals)

        age_grid = np.empty((n_ages, n_metal))
        metal_grid = np.empty((n_ages, n_metal))

        # Here we make sure the spectra are sorted in both [M/H] and Age
        # along the two axes of the rectangular grid of templates.
        for j, age in enumerate(ages):
            for k, metal in enumerate(metals):
                age_grid[j, k] = age
                metal_grid[j, k] = metal

        self.age_grid = age_grid
        self.metal_grid = metal_grid
        self.n_ages = n_ages
        self.n_metal = n_metal


def age_metal_alpha(filename):
    """
    Extract the age, alpha and metallicity from the name of a file of
    the MILES library of Single Stellar Population models as
    downloaded from http://miles.iac.es/ as of 2016

    :param filename: string possibly including full path
        (e.g. 'miles_library/Mun1.30Zm0.40T03.9811.fits')
    :return: age (Gyr), [M/H]

    """

    s = path.basename(filename)
    age = float(s[s.find('T')+1:s.find('_')])
    metal = s[s.find('Z')+1:s.find('T')]
    alpha = float(s[s.find('.fits')-2:s.find('.fits')])
    if "m" in metal:
        metal = -float(metal[1:])
    elif "p" in metal:
        metal = float(metal[1:])
    else:
        raise ValueError("This is not a standard MILES filename")

    return age, metal, alpha


def age_metal_abun_var(filename, prefix='C'):
    """
    Extract the age, alpha and metallicity from the name of a file of
    the MILES library of Single Stellar Population models as
    downloaded from http://miles.iac.es/ as of 2016

    :param filename: string possibly including full path
        (e.g. 'miles_library/Mun1.30Zm0.40T03.9811.fits')
    :return: age (Gyr), [M/H]

    """

    s = path.basename(filename)
    age = float(s[s.find('T')+1:s.find('_')])
    metal = s[s.find('Z')+1:s.find('T')]
    abun = s[s.find(prefix)+len(prefix):s.find('.fits')]
    if "m" in metal:
        metal = -float(metal[1:])
    elif "p" in metal:
        metal = float(metal[1:])
    if "m" in abun:
        abun = -float(abun[1:])
    elif "p" in abun:
        abun = float(abun[1:])
    else:
        raise ValueError("This is not a standard MILES filename")

    return age, metal, abun


class miles_abun_var(object):

    def __init__(self, pathname, velscale, FWHM_gal,
                 FWHM_tem=2.51, normalize=False, age_lim=None, prefix='C', metal_lim=None, abun_lim=None,
                 instrument=None, wavelength_unit=u.AA, ssp_model_label='alpha'):

        files = glob.glob(pathname)
        assert len(files) > 0, "Files not found %s" % pathname

        if prefix != 'alpha':
            all = [age_metal_abun_var(f, prefix=prefix) for f in files]
        else:
            all = [age_metal_alpha(f) for f in files]
        all_ages, all_metals, all_abuns = np.array(all).T
        ages, metals, abuns = np.unique(all_ages), np.unique(all_metals), np.unique(all_abuns)

        assert set(all) == set([(a, b, c) for a in ages for b in metals for c in abuns]), \
            'Ages and Metals do not form a Cartesian grid'

        # Extract the wavelength range and logarithmically rebin one spectrum
        # to the same velocity scale of the SDSS galaxy spectrum, to determine
        # the size needed for the array which will contain the template spectra.
        hdu = fits.open(files[0])
        ssp = hdu[0].data
        h2 = hdu[0].header
        lam_range_temp = h2['CRVAL1'] + np.array([0, h2['CDELT1']*(h2['NAXIS1']-1)])

        ssp_wave = np.linspace(lam_range_temp[0], lam_range_temp[-1], len(ssp))
        mask = (ssp_wave > 3000) & (ssp_wave < 2.5e4)
        ssp_wave = ssp_wave[mask]
        ssp = ssp[mask]

        sspNew, log_lam_temp = util.log_rebin(
            [ssp_wave[0], ssp_wave[-1]], ssp, velscale=velscale)[:2]

        # Choose templates with age limit!
        if not age_lim is None:
            if np.isscalar(age_lim):
                ages = ages[ages >= age_lim]
            elif len(age_lim) == 2:
                mask_age = (ages >= age_lim[0]) & (ages < age_lim[1])
                ages = ages[mask_age]
            else:
                print('Wrong input for age_lim!')
        if not metal_lim is None:
            if np.isscalar(metal_lim):
                metals = metals[metals >= metal_lim]
            elif len(metal_lim) == 2:
                mask_metals = (metals >= metal_lim[0]) & (metals < metal_lim[1])
                metals = metals[mask_metals]
            else:
                print('Wrong input for age_lim!')
        if not abun_lim is None:
            abuns = abuns[abuns >= abun_lim]

        n_ages = len(ages)
        n_metal = len(metals)
        n_abuns = len(abuns)

        templates = np.empty((sspNew.size, n_ages, n_metal, n_abuns))
        age_grid = np.empty((n_ages, n_metal, n_abuns))
        metal_grid = np.empty((n_ages, n_metal, n_abuns))
        abun_grid = np.empty((n_ages, n_metal, n_abuns))

        # Convolve the whole Vazdekis library of spectral templates
        # with the quadratic difference between the galaxy and the
        # Vazdekis instrumental resolution. Logarithmically rebin
        # and store each template as a column in the array TEMPLATES.

        # Quadratic sigma difference in pixels Vazdekis --> galaxy
        # The formula below is rigorously valid if the shapes of the
        # instrumental spectral profiles are well approximated by Gaussians.
        FWHM_dif = get_FWHM_diff(ssp_wave)

        sigma = FWHM_dif/2.355/h2['CDELT1']  # sigma in template pixels

        # Here we make sure the spectra are sorted in both [M/H] and Age
        # along the two axes of the rectangular grid of templates.
        for j, age in enumerate(ages):
            for k, metal in enumerate(metals):
                for l, abun in enumerate(abuns):
                    p = all.index((age, metal, abun))
                    hdu = fits.open(files[p])
                    ssp = hdu[0].data
                    ssp = ssp[mask]
                    if np.isscalar(sigma):
                        ssp = ndimage.gaussian_filter1d(ssp, sigma)
                    else:
                        ssp = util.gaussian_filter1d(ssp, sigma)  # convolution with variable sigma
                    sspNew = util.log_rebin([ssp_wave[0], ssp_wave[-1]], ssp, velscale=velscale)[0]
                    if normalize:
                        sspNew /= np.mean(sspNew)  # changes to luminosity weighted
                    # K. FAH: ADDED THIS LINE FOR THE EMILES TEMPLATES
                    sspNew[np.isnan(sspNew)] = 0

                    templates[:, j, k, l] = sspNew
                    age_grid[j, k, l] = age
                    metal_grid[j, k, l] = metal
                    abun_grid[j, k, l] = abun

        self.templates = templates/np.median(templates)  # Normalize by a scalar, mass weighted SSPs
        self.log_lam_temp = log_lam_temp
        self.ssp_model_label = ssp_model_label
        self.age_grid = age_grid
        self.metal_grid = metal_grid
        self.abun_grid = abun_grid
        self.n_ages = n_ages
        self.n_metal = n_metal
        self.n_abuns = n_abuns
        self.ages = ages
        self.metals = metals
        self.abuns = abuns
        self.velscale = velscale
        self.wavelength_unit = wavelength_unit
        


class sinfoni_k_stars(object):

    def __init__(self, pathname, velscale, instrument=None, wavelength_unit=u.micron, ssp_model_label='sinfoni_k'):

        files = glob.glob(pathname)
        assert len(files) > 0, "Files not found %s" % pathname

        if instrument != 'SINFONI':
            print('Should not use these spectra if not fitting SINFONI data!')
            
        hdu = fits.open(files[0])
        ssp = hdu[0].data
        h2 = hdu[0].header
        lam_range_temp = (h2['CRVAL1'] + np.array([0, h2['CDELT1']*(h2['NAXIS1']-1)])) #in micron

        ssp_wave = np.linspace(lam_range_temp[0], lam_range_temp[-1], len(ssp))
        mask = (ssp_wave > 2.0) & (ssp_wave < 2.5)
        ssp_wave = ssp_wave[mask]
        ssp = ssp[mask]

        sspNew, log_lam_temp = util.log_rebin(
            [ssp_wave[0], ssp_wave[-1]], ssp, velscale=velscale)[:2]

        templates = np.empty((sspNew.size, len(files)))

        for i in range(len(files)):
            hdu = fits.open(files[i])
            ssp = hdu[0].data
            ssp = ssp[mask]
            sspNew = util.log_rebin([ssp_wave[0], ssp_wave[-1]], ssp, velscale=velscale)[0]
            sspNew[np.isnan(sspNew)] = 0
            templates[:, i] = sspNew

        self.templates = templates/np.median(templates)  # Normalize by a scalar, mass weighted SSPs
        self.log_lam_temp = log_lam_temp
        self.ssp_model_label = ssp_model_label
        self.velscale = velscale
        self.wavelength_unit = wavelength_unit
        
        
        

#%%%%%%%%%%%%%%%%%%% XSL

def age_metal_xsl(filename):
    hdr = fits.getheader(filename)
    metal = float(hdr['MH'])
    logage = float(hdr['LOGAGE'])
    age = 10**(logage - 9)
    age = np.round(age, 2)
    return age, metal


class xsl_ssp_models(object):
    def __init__(self, pathname, instrument=None, wavelength_unit=u.micron, ssp_model_label='XSL'):

        files = glob.glob(pathname)
        assert len(files) > 0, "Files not found %s" % pathname
        
        #get all ages and metallicities
        all = [age_metal_xsl(file) for file in files]
        
        all_ages, all_metals = np.array(all).T
        ages, metals = np.unique(all_ages), np.unique(all_metals)
        
        assert set(all) == set([(a, b) for a in ages for b in metals]), \
            'Ages and Metals do not form a Cartesian grid'

        #read the first file
        hdu = fits.open(files[0])
        hdr = hdu[0].header
        ssp = hdu[0].data
        
        ssp_wave =  10**((np.arange(len(ssp)) - hdr['CRPIX1']) * hdr['CDELT1'] + hdr['CRVAL1'])/1e3 #from nm to micron 
        
        #mask down to the IR part (K-band in this case) only
        if instrument == 'SINFONI':
            mask = (ssp_wave > 2.0) & (ssp_wave < 2.8)
            ssp_wave = ssp_wave[mask]
            sspNew = ssp[mask]     
        if instrument == 'MUSE':
            ssp_wave = ssp_wave * 1e4 #to AA
            mask = (ssp_wave > 4500) & (ssp_wave < 9500)
            ssp_wave = ssp_wave[mask]
            sspNew = ssp[mask]                
            wavelength_unit = u.AA
            
        #xsl is already log10 rebinned, but the lams are in linear units
        #convert to ln
        log_lam_ssp = np.log10(ssp_wave) #this one has now a regular spacing 
        ln_lam_ssp = log_lam_ssp * np.log(10) #this is to convert to lns
        lam_ssp = np.exp(ln_lam_ssp) #this is now going back to linear units, but from the ln

        d_ln_lam_ssp = np.diff(ln_lam_ssp[[0, -1]])/(ln_lam_ssp.size -1) 
        c = 299792.458                              # speed of light in km/s
        velscale = c*d_ln_lam_ssp
        velscale = velscale.item()

        #pixel size in micron of every pixel, needed for convolution
        dlam_ssp = np.diff(lam_ssp) #size of every pixel in micron
        dlam_ssp = np.append(dlam_ssp, dlam_ssp[-1]) #back to original length of array
        #now change resolution to match SINFONI
        
        #xsl has a velocity resolution (sigma) of 16 km/s
        #FWHM = 2.355*sigma #in velocities
        #convert this to pix
        sig = 16
        fwhm_lam_ssp = sig/c * lam_ssp * 2.355 #fwhm_ssp in micron

        #sinfoni has sig = 33 #from above (2.45 AA per pixel dispersion according to some paper)
        if instrument == 'SINFONI':
            sig_gal = 35
            fwhm_lam_gal = sig_gal/c * lam_ssp * 2.355 # as fwhm
            
            R = 4000
            fwhm_lam_gal = lam_ssp/R
        if instrument == 'MUSE':
            fwhm_lam_gal = get_FWHM_MUSE(lam_ssp) #in AA
            
        #eris probably as has less... 

        #difference first in micron, then in pixel for the convolution
        fwhm_diff = np.sqrt((fwhm_lam_gal**2) - fwhm_lam_ssp**2)  #in micron  
        sigma = fwhm_diff/2.355/(dlam_ssp) #in pixel #wavelength dependent
        
        #now create the templates array
        n_ages = len(ages)
        n_metal = len(metals)
        
        templates = np.empty((sspNew.size, n_ages, n_metal))
        age_grid = np.empty((n_ages, n_metal))
        metal_grid = np.empty((n_ages, n_metal))
        
        for j, age in enumerate(ages):
            for k, metal in enumerate(metals):
                p = all.index((age, metal))
                hdu = fits.open(files[p])            
                sspNew = hdu[0].data  
                sspNew = sspNew[mask]
                
                
                sspNew = util.gaussian_filter1d(sspNew, sigma)  # convolution with variable sigma
                
                sspNew[np.isnan(sspNew)] = 0
                templates[:, j, k] = sspNew
                age_grid[j, k] = age
                metal_grid[j, k] = metal

        self.templates = templates/np.median(templates)  # Normalize by a scalar, mass weighted SSPs
        self.log_lam_temp = ln_lam_ssp
        self.wave_linear = lam_ssp
        self.metal_grid = metal_grid
        self.age_grid = age_grid
        self.velscale = velscale
        self.wavelength_unit = wavelength_unit
        self.n_ages = n_ages
        self.n_metal = n_metal
        self.ssp_model_label = ssp_model_label
        
    def mean_age_metal(self, weights, quiet=False):

        assert self.age_grid.shape == self.metal_grid.shape == weights.shape, \
            "Input weight dimensions do not match"

        log_age_grid = np.log10(self.age_grid) + 9
        metal_grid = self.metal_grid

        # These are eq.(1) and (2) in McDermid+15
        # http://adsabs.harvard.edu/abs/2015MNRAS.448.3484M
        mean_log_age = np.sum(weights*log_age_grid)/np.sum(weights)
        mean_metal = np.sum(weights*metal_grid)/np.sum(weights)

        if not quiet:
            print('Weighted <logAge> [yr]: %.3g' % mean_log_age)
            print('Weighted <[M/H]>: %.3g' % mean_metal)

        return mean_log_age, mean_metal
