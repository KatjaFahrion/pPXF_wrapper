###############################################################################
#
# Many of these util functions are inspired and adapted from the code in ppxf (miles_util.py)
#
############################################################################################

import numpy as np
from scipy import ndimage
from astropy.io import fits
import ppxf.ppxf_util as util
import ppxf.ppxf as ppxf
import ppxf.miles_util as miles_util
import glob
import astropy.units as u
import matplotlib.pyplot as plt

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


def get_FWHM_diff(lam, lam_lim=10000, fwhm=-1): #only for EMILES
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
                

class ssp_templates(object):

    def __init__(self, pathname, velscale=0, fwhm_gal=-1, normalize=False, 
                 age_lim=None, metal_lim=None, instrument=None, wavelength_unit=u.AA, 
                 ssp_model_label='EMILES'):
        
        self.instrument = instrument
        self.ssp_model_label = ssp_model_label
        self.fwhm_gal = fwhm_gal
        self.normalize = normalize
        self.wavelength_unit = wavelength_unit
        self.age_lim = age_lim
        self.metal_lim = metal_lim
        self.velscale = velscale
        self.pathname = pathname
        
        files = glob.glob(self.pathname)
        if not len(files) > 0:
            'No SSP templates found!'
            return

        if ('MILES' in self.ssp_model_label) or (self.ssp_model_label == 'XSL'):
            self.all = [self.get_age_metal(f) for f in files]
            all_ages, all_metals = np.array(self.all).T
            ages, metals = np.unique(all_ages), np.unique(all_metals)
            
            assert set(self.all) == set([(a, b) for a in ages for b in metals]), \
                'Ages and Metals do not form a Cartesian grid'
            #apply age and metal lims
            if not self.age_lim is None:
                if np.isscalar(self.age_lim):
                    ages = ages[ages >= self.age_lim]
                elif len(self.age_lim) == 2:
                    mask_age = (ages >= self.age_lim[0]) & (ages < self.age_lim[1])
                    ages = ages[mask_age]
                else:
                    print('Wrong input for age_lim!')
            if not self.metal_lim is None:
                if np.isscalar(self.metal_lim):
                    metals = metals[metals >= self.metal_lim]
                elif len(self.metal_lim) == 2:
                    mask_metals = (metals >= self.metal_lim[0]) & (metals < self.metal_lim[1])
                    metals = metals[mask_metals]
                else:
                    print('Wrong input for metal_lim!')
                    
            self.ages = ages
            self.metals = metals
            
        
        if 'MILES' in self.ssp_model_label:
            self.get_templates_miles(files)
        
        elif self.ssp_model_label == 'XSL':
            self.get_templates_xsl(files)
        
        elif self.ssp_model_label == 'sinfoni_k':
            self.get_templates_sinfoni_k(files)
        
        
    def get_FWHM_diff(self, lam):
        """
        Function to get the FWHM difference between models and spectrum
        """
        c = 299792.458      #km/s
        if self.instrument == 'MUSE':
            #assume AA
            if self.fwhm_gal == -1: #assume the standard fwhm from Gueron:
                fwhm_lam_gal = get_FWHM_MUSE(lam)
            else:
                fwhm_lam_gal = np.full_like(lam, self.fwhm_gal)
        
        if self.instrument == 'OSIRIS':
            #assume AA
            if self.fwhm_gal == -1: 
                R = 1000
                fwhm_lam_gal = np.full_like(lam, lam/R)
            else:
                fwhm_lam_gal = np.full_like(lam, self.fwhm_gal)
                
        
        if self.instrument in ['MUSE', 'OSIRIS']:        
            if self.ssp_model_label in ['alpha', 'MILES_solar', 'MILES_alpha', 'EMILES']:
                #MILES has 2.51 AA fwhm
                self.fwhm_lam_ssp = np.full_like(lam, 2.51) #AA
                
            if self.ssp_model_label == 'XSL':
                sig = 16
                self.fwhm_lam_ssp = sig/c * lam * 2.355 #same unit as lam
                
            
        if self.instrument == 'SINFONI':
            #assume micron
            if self.fwhm_gal == -1:
                #assume R = 4000
                R = 4000
                fwhm_lam_gal = lam/R
            else:
                fwhm_lam_gal = np.full_like(lam, self.fwhm_gal)
            
            if self.ssp_model_label in ['alpha', 'MILES_solar', 'MILES_alpha', 'EMILES']:
                #MILES sig = 60 km/s in SINFONI range
                sig = 60
                self.fwhm_lam_ssp = sig/c * lam * 2.355 #same unit as lam
            if self.ssp_model_label == 'XSL':
                sig = 16
                self.fwhm_lam_ssp = sig/c * lam * 2.355 #same unit as lam
        
        self.fwhm_lam_gal = fwhm_lam_gal
        
        
        if np.nanmean(self.fwhm_lam_gal) > np.nanmean(self.fwhm_lam_ssp):
            #this is how it should be, the galaxy spectrum should have a lower resolution
            self.broaden_gal = False
            self.broaden_ssp = True
            #mask any regions where it might not be the case
            mask_low = self.fwhm_lam_gal < self.fwhm_lam_ssp
            self.fwhm_lam_gal[mask_low] = self.fwhm_lam_ssp[mask_low] #to get zero difference here
            #self.fwhm_lam_gal = fwhm_lam_gal
            self.FWHM_diff = np.sqrt(self.fwhm_lam_gal**2 - self.fwhm_lam_ssp**2)
        else:
            print('Spectrum has higher resolution than models!! Need to convolve to match models')
            self.broaden_gal = True
            self.broaden_ssp = False
            mask_low = self.fwhm_lam_gal > self.fwhm_lam_ssp
            self.fwhm_lam_ssp[mask_low] = self.fwhm_lam_gal[mask_low]
            self.FWHM_diff = np.sqrt(self.fwhm_lam_ssp**2 - self.fwhm_lam_gal**2)
        
        
    def get_age_metal(self, filename):
        """_summary_
        Get age and metal 
        """
        if 'MILES' in self.ssp_model_label:
            #use the ppxf miles_util:
            age, metal = miles_util.age_metal(filename)
        if self.ssp_model_label == 'XSL':
            hdr = fits.getheader(filename)
            metal = float(hdr['MH'])
            logage = float(hdr['LOGAGE'])
            age = 10**(logage - 9)
            age = np.round(age, 2)
        return age, metal
    
    
    def get_templates_miles(self, files):
        ###
        # Read MILES templates
        # read first file to get the dimensions
        hdu = fits.open(files[0])
        ssp = hdu[0].data
        hdr = hdu[0].header
        lam_range_temp = hdr['CRVAL1'] + np.array([0, hdr['CDELT1']*(hdr['NAXIS1']-1)]) #in AA

        ssp_wave = np.linspace(lam_range_temp[0], lam_range_temp[-1], len(ssp))
        
        if self.instrument == 'MUSE':
            mask = (ssp_wave > 4300) & (ssp_wave < 9800)
            self.wavelength_unit = u.AA
        
        if self.instrument == 'SINFONI':
            self.wavelength_unit = u.micron
            ssp_wave = ssp_wave / 1e4 #to micron
            mask = (ssp_wave > 1.9) & (ssp_wave < 2.6)
        
        if self.instrument == 'OSIRIS':
            mask = (ssp_wave > 3000) & (ssp_wave < 7800)
            self.wavelength_unit = u.AA
        
        ssp_wave = ssp_wave[mask]
        ssp = ssp[mask]

        sspNew, log_lam_temp = util.log_rebin(
            [ssp_wave[0], ssp_wave[-1]], ssp, velscale=self.velscale)[:2]

        self.n_ages = len(self.ages)
        self.n_metal = len(self.metals)

        templates = np.empty((sspNew.size, self.n_ages, self.n_metal))
        templates_lin = np.empty((ssp_wave.size, self.n_ages, self.n_metal)) #templates linear
        age_grid = np.empty((self.n_ages, self.n_metal))
        metal_grid = np.empty((self.n_ages, self.n_metal))
        
        #get the FWHM difference between the ssp templates and the galaxy
        self.get_FWHM_diff(ssp_wave) #in AA or micron, depending on instrument
        
        dpix = ssp_wave[1] - ssp_wave[0]
        sigma = self.FWHM_diff/2.355/dpix  # sigma in pixels of MILES templates

        #populate the templates array and the age and metallicity grids
        for j, age in enumerate(self.ages):
            for k, metal in enumerate(self.metals):
                p = self.all.index((age, metal))
                hdu = fits.open(files[p])
                ssp = hdu[0].data
                ssp = ssp[mask]
                if self.broaden_ssp:
                    if np.isscalar(sigma):
                        ssp = ndimage.gaussian_filter1d(ssp, sigma)
                    else:
                        ssp = util.gaussian_filter1d(ssp, sigma)  # convolution with variable sigma
                sspNew = util.log_rebin([ssp_wave[0], ssp_wave[-1]], ssp, velscale=self.velscale)[0]

                if self.normalize:
                    sspNew /= np.nanmean(sspNew)  # changes to luminosity weighted
                sspNew[np.isnan(sspNew)] = 0
                templates_lin[:, j, k] = ssp
                templates[:, j, k] = sspNew
                age_grid[j, k] = age
                metal_grid[j, k] = metal
        
        self.templates = templates/np.nanmedian(templates)  
        self.templates_lin = templates_lin
        self.age_grid = age_grid
        self.metal_grid = metal_grid  
        self.wave_linear = ssp_wave
        self.log_lam_temp = log_lam_temp
    

    def get_templates_xsl(self, files):
        # to read XSL SSP templates
        #read the first file
        hdu = fits.open(files[0])
        hdr = hdu[0].header
        ssp = hdu[0].data
        
        ssp_wave =  10**((np.arange(len(ssp)) - hdr['CRPIX1']) * hdr['CDELT1'] + hdr['CRVAL1'])/1e3 #from nm to micron 
        
        #mask down to the IR part (K-band in this case) only
        if self.instrument == 'SINFONI':
            mask = (ssp_wave > 2.0) & (ssp_wave < 2.8)
            ssp_wave = ssp_wave[mask]
            sspNew = ssp[mask]     
        if self.instrument == 'MUSE':
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
        self.velscale = velscale

        #pixel size in micron of every pixel, needed for convolution
        dlam_ssp = np.diff(lam_ssp) #size of every pixel in micron
        dlam_ssp = np.append(dlam_ssp, dlam_ssp[-1]) #back to original length of array
        #now change resolution to match SINFONI
        
        self.get_FWHM_diff(ssp_wave) #in AA or micron, depending on instrument

        sigma = self.FWHM_diff/2.355/(dlam_ssp) #in pixel #wavelength dependent
        
        #now create the templates array
        self.n_ages = len(self.ages)
        self.n_metal = len(self.metals)
        
        templates = np.empty((sspNew.size, self.n_ages, self.n_metal))
        age_grid = np.empty((self.n_ages, self.n_metal))
        metal_grid = np.empty((self.n_ages, self.n_metal))
        
        for j, age in enumerate(self.ages):
            for k, metal in enumerate(self.metals):
                p = self.all.index((age, metal))
                hdu = fits.open(files[p])            
                sspNew = hdu[0].data  
                sspNew = sspNew[mask]
                
                if self.broaden_ssp:
                    sspNew = util.gaussian_filter1d(sspNew, sigma)  # convolution with variable sigma
                if self.normalize:
                    sspNew /= np.nanmean(sspNew)
                sspNew[np.isnan(sspNew)] = 0
                
                templates[:, j, k] = sspNew
                age_grid[j, k] = age
                metal_grid[j, k] = metal

        self.templates = templates/np.nanmedian(templates)  
        self.log_lam_temp = ln_lam_ssp
        self.wave_linear = lam_ssp
        self.metal_grid = metal_grid
        self.age_grid = age_grid
        self.velscale = velscale
        
        
    def get_templates_sinfoni_k(self, files):
        ## Read sinfoni_k stars
        
        if self.instrument != 'SINFONI':
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
            [ssp_wave[0], ssp_wave[-1]], ssp, velscale=self.velscale)[:2]

        templates = np.empty((sspNew.size, len(files)))

        for i in range(len(files)):
            hdu = fits.open(files[i])
            ssp = hdu[0].data
            ssp = ssp[mask]
            sspNew = util.log_rebin([ssp_wave[0], ssp_wave[-1]], ssp, velscale=self.velscale)[0]
            sspNew[np.isnan(sspNew)] = 0
            templates[:, i] = sspNew

        self.templates = templates/np.nanmedian(templates)  # Normalize by a scalar to avoid numerical issues
        self.log_lam_temp = log_lam_temp
        self.wave_linear = ssp_wave
        self.broaden_gal = False
        self.broaden_ssp = False
        



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



class xsl_ssp_models(object):
    def __init__(self, pathname, instrument=None, wavelength_unit=u.micron, ssp_model_label='XSL'):

        files = glob.glob(pathname)
        assert len(files) > 0, "Files not found %s" % pathname
        
        #get all ages and metallicities
        all = [get_age_metal(file) for file in files]
        
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
