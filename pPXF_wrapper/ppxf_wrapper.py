"""
Routines to fit the MUSE spectra

Author: Katja Fahrion
"""


from __future__ import print_function

from os import path
import matplotlib.pylab as plt

import numpy as np
import ppxf.ppxf_util as util
import ppxf.ppxf as ppxf
from os import path

# modified version of miles_util.py, changed to read in the EMILES templates
from . import templates_util as lib
from . import ppxf_wrapper_support as sup  # some supporting functions
from . import ppxf_wrapper_plotting_routines as plotting  # some plotting routines

import multiprocessing as mp
from functools import partial

import warnings
warnings.filterwarnings("ignore")


# %%

class ppwrapper():
    """ 
    Class that encompasses the fit options, templates and results
    
    """
    def __init__(self, wave, spec_lin, noise_spec=None, fwhm=-1, galaxy='FCC47', vel=1366, sig=127, h3=None, h4=None, plot=True, kin_only=True,
              moments=2, degree=12, mdegree=8, regul=0, quiet=True,  lam_range=[4700, 2.4e4], plot_kin_title='Spectrum_kin_fit.png', 
              plot_pop_title='Spectrum_pop_fit.png',  plot_out='./', save_plots=False, templates_path=None, 
              age_lim=None, metal_lim=None, abun_fit=False, mask_file=None, 
              logbin=True, velscale=None, gas_fit=False, no_kin_fit=False, start=[], light_weighted=False, templates=None,
              instrument = 'MUSE', ssp_models = 'EMILES', velscale_ratio=1, pp=None, Spec=None, normalize=True, age = None,
              metal = None, abun=None, dage = None, dmetal=None, MC=False, n=1, cores=1, filebase_MC = 'fit_MC', out_dir = './', plot_hist=False,
              savetxt = True, v_mc=None, dv_mc=None, sig_mc=None, dsig_mc=None, h3_mc=None, dh3_mc=None, h4_mc=None, dh4_mc=None, 
              age_mc=None, dage_mc=None, metal_mc=None, dmetal_mc=None, abun_mc=None, dabun_mc=None, v_gas_mc=None, 
              dv_gas_mc=None, sig_gas_mc = None, dsig_gas_mc=None):
        
        self.wave = wave
        self.spec_lin = spec_lin
        self.noise_spec = noise_spec
        self.fwhm = fwhm
        self.galaxy = galaxy
        self.vel = vel
        self.sig = sig
        self.plot = plot
        self.kin_only = kin_only
        self.moments = moments
        self.degree = degree
        self.mdegree = mdegree
        self.regul = regul
        self.quiet = quiet
        self.lam_range = lam_range
        self.plot_kin_title = plot_kin_title
        self.plot_pop_title = plot_pop_title
        self.plot_out = plot_out
        self.save_plots = save_plots
        #templates
        self.templates_path = templates_path
        self.age_lim = age_lim
        self.metal_lim = metal_lim
        self.mask_file = mask_file
        self.abun_fit = abun_fit
        self.logbin = logbin
        self.velscale = velscale
        self.gas_fit = gas_fit
        self.no_kin_fit = no_kin_fit
        self.start = start
        self.light_weighted = light_weighted
        self.templates = templates
        self.instrument = instrument
        self.ssp_models = ssp_models
        self.velscale_ratio = velscale_ratio
        #results
        self.pp = pp
        self.Spec = Spec
        self.normalize = normalize
        self.age = age
        self.dage = dage
        self.metal = metal
        self.dmetal = dmetal
        self.abun = abun
        self.h3 = h3
        self.h4 = h4
        #MC stuff
        self.MC = MC
        self.n = n
        self.cores = cores
        self.filebase_MC = filebase_MC
        self.out_dir = out_dir
        self.plot_hist = plot_hist
        self.savetxt = savetxt
        #MC results
        self.v_mc = v_mc
        self.dv_mc = dv_mc
        self.sig_mc = sig_mc
        self.dsig_mc = dsig_mc
        self.age_mc = age_mc
        self.dage_mc = dage_mc
        self.metal_mc = metal_mc
        self.dmetal_mc = dmetal_mc
        self.h3_mc = h3_mc
        self.dh3_mc = dh3_mc
        self.h4_mc = h4_mc
        self.dh4_mc = dh4_mc
        self.abun_mc = abun_mc
        self.dabun_mc = dabun_mc
        self.v_gas_mc = v_gas_mc
        self.dv_gas_mc = dv_gas_mc
        self.sig_gas_mc = sig_gas_mc
        self.dsig_gas_mc = dsig_gas_mc
        
        
    #plotting functions
    def plot_kin_fit(self, text_loc=[0.98, 0.4], ax=None, xlabel=None, ylabel=None,
                           title=None, legend_loc='upper left',
                           show_sig=True):
        plotting.plot_kin_fit(self, text_loc=text_loc, ax=ax, xlabel=xlabel, ylabel=ylabel, title=title, 
                              legend_loc=legend_loc, show_sig=show_sig)
    
    def plot_pop_fit(self, ax0=None, ax1=None, xlabel=None, ylabel=None,
                           title=None, legend_loc='upper left', zoom_to_stars = False):
        plotting.plot_pop_fit(self, ax0=ax0, ax1=ax1, xlabel=xlabel, ylabel=ylabel, zoom_to_stars=zoom_to_stars, 
                              title=title, legend_loc=legend_loc)
        
    def plot_age_metal_grid(self, ax=None,  colorbar_position='top',
                         grid_plot_title='age_metal_grid', outdir=None):
        plotting.plot_age_metal_grid(self, ax=ax, colorbar_position=colorbar_position, outdir=outdir, grid_plot_title=grid_plot_title)
    
    def initialize_spectrum_and_templates(self, dir):
        """Function to get spectra and templates in correct formats

        Args:
            dir (string): path to base of templates
        """
        if self.logbin:
            Spec = Spectrum(self.wave, self.spec_lin, lam_range=self.lam_range,
                            galaxy=self.galaxy, spec_noise_lin=self.noise_spec, 
                            instrument=self.instrument, velscale_ratio=self.velscale_ratio)
        else:
            Spec = Spectrum(self.wave, self.spec_lin, lam_range=self.lam_range,
                            galaxy=self.galaxy, spec_log=self.spec_lin, logLam=self.wave, 
                            velscale=self.velscale, spec_noise_log=self.noise_spec, 
                        instrument=self.instrument, velscale_ratio=self.velscale_ratio)
        self.Spec = Spec
        
        if self.light_weighted:
            self.normalize = True
        else:
            self.normalize = False

        if self.templates is None:
            #get the templates
            #check if requested templates are supported
            available_models = ['EMILES', 'XSL', 'sinfoni_k', 'alpha', 'MILES_solar', 'MILES_alpha']
            if not self.ssp_models in available_models:
                print("{0} models not available, try: {1}".format(self.ssp_models, available_models))
                return 
            if self.instrument == 'SINFONI':
                if not self.ssp_models in ['XSL', 'sinfoni_k', 'EMILES']:
                    print('{0} not available for instrument {1}, use {2}'.format(self.ssp_models, self.instrument, 
                                                                                ['XSL', 'sinfoni_k', 'EMILES']))
                    return
            if self.instrument == 'NIRSPEC':
                if not self.ssp_models in ['XSL', 'sinfoni_k', 'EMILES']:
                    print('{0} not available for instrument {1}, use {2}'.format(self.ssp_models, self.instrument, 
                                                                                ['XSL', 'sinfoni_k', 'EMILES']))
                    return
            
            if self.templates_path is None:
                if self.ssp_models == 'EMILES':
                    self.templates_path = dir + '/EMILES_Basti/Ebi1.30*.fits'
                if self.ssp_models == 'MILES_solar':
                    self.templates_path = dir + '/MILES_scaled_solar/Mbi1.30*.fits'
                if self.ssp_models == 'MILES_alpha':
                    self.templates_path = dir + '/MILES_alpha_enhanced/Mbi1.30*.fits'

            
                if self.ssp_models == 'alpha':
                    self.abun_fit = True
                    self.templates_path = dir + '/BastiAlpha_all/Mbi*.fits'
                    if self.lam_range[1] > 7100:
                        self.lam_range = [4500, 7100]
                    if not self.quiet:
                        print('Alpha-variable fit!')
                        print('This will take a while!')
                    
                if self.ssp_models == 'sinfoni_k':
                    self.templates_path = dir + 'sinfoni_k/*.fits'
                    
                if self.ssp_models == 'XSL':
                    self.templates_path = dir + '/XSL_SSP_Kroupa/XSL*.fits'
                
                self.templates = lib.ssp_templates(self.templates_path, velscale=self.Spec.velscale, fwhm_gal=self.fwhm,
                                                   age_lim = self.age_lim, metal_lim = self.metal_lim, normalize = self.normalize,
                                                   instrument = self.instrument, ssp_model_label = self.ssp_models)

        else:
            self.ssp_models = self.templates.ssp_model_label
            
        #if XSL, repeat the rebinning
        if self.ssp_models == 'XSL':
            
            if (self.instrument == 'SINFONI') or (self.instrument == 'NIRSPEC'):
                #redo the binning in that case
                self.velscale_ratio = 3
                self.velscale = 3*self.templates.velscale
                        
            if (self.instrument == 'MUSE') or (self.instrument == 'OSIRIS'):
                self.velscale_ratio = 5
                self.velscale = 5*self.templates.velscale
                        
            self.Spec = Spectrum(self.wave, self.spec_lin, lam_range=self.lam_range,
                galaxy=self.galaxy, spec_noise_lin=self.noise_spec, 
                instrument=self.instrument, velscale_ratio=self.velscale_ratio, velscale=self.velscale)
            
        #if the spectrum has a lower resolution than the templates, need to convolve it (e.g. SINFONI fitted with EMILES)
        if self.templates.broaden_gal:
            dlam = self.wave[1] - self.wave[0]
            #redo the FWHM_diff calculation
            self.templates.get_FWHM_diff(self.wave)
            
            sigma = self.templates.FWHM_diff/2.355/(dlam) #in pixel #wavelength dependent
            spec_lin_new = util.gaussian_filter1d(self.spec_lin, sigma)
            mask = ~np.isfinite(spec_lin_new)
            spec_lin_new[mask] = 0
            #redo the logbinning in this case
            self.Spec = Spectrum(self.wave, spec_lin_new, lam_range=self.lam_range,
                galaxy=self.galaxy, spec_noise_lin=self.noise_spec, 
                instrument=self.instrument, velscale_ratio=self.velscale_ratio, velscale=self.templates.velscale)
    
    def get_age_metal(self):
        if self.pp is None:
            print('Need pp object!')
            return
        #get weighted ages and metallicities from a given set of weights
        weights = self.pp.weights[~self.pp.gas_component].reshape(
            self.templates.n_ages, self.templates.n_metal)/self.pp.weights[~self.pp.gas_component].sum()

        xgrid = self.templates.age_grid
        ygrid = self.templates.metal_grid
        mean_age = np.sum(weights*xgrid)/np.sum(weights)
        mean_metal = np.sum(weights*ygrid)/np.sum(weights)

        std_age = np.sqrt(np.sum(weights)/(np.sum(weights)**2 - np.sum(weights**2))
                            * np.sum(weights * (xgrid - mean_age)**2))
        std_metal = np.sqrt(np.sum(weights)/(np.sum(weights)**2 - np.sum(weights**2))
                                * np.sum(weights * (ygrid - mean_metal)**2))
        
        self.age = mean_age
        self.metal = mean_metal
        self.dage = std_age
        self.dmetal = std_metal
        return mean_age, mean_metal, std_age, std_metal

    

class Spectrum():
    '''
    Spectrum class to simplify things,
    doing the log_rebinning if needed (e.g. for MUSE and SINFONI spectra)
    '''

    def __init__(self, wave_lin, spec_lin, lam_range=[4000, 2.5e4], spec_noise_lin=None, galaxy='FCC47', spec_log=None,
                 logLam=None, velscale=None, noise_val=1, instrument='MUSE', velscale_ratio = 1):
        self.spec_lin = spec_lin
        self.wave_lin = wave_lin
        self.lam_range = lam_range
        self.spec_noise_lin = spec_noise_lin
        self.galaxy = galaxy
        self.AO = False
        self.instrument = instrument
        self.log_median_value = 0
        self.noise_val = noise_val
        self.velscale = velscale
        self.velscale_ratio = velscale_ratio #usually 1 (meaing the templates have the same velscale, except for XSL)

        # Do the binning
        if spec_log is None:
            self.log_bin()
        else:
            self.spec_log = spec_log/np.nanmedian(spec_log) #first normalize the spectrum
            self.logLam = logLam
            self.velscale = velscale
            if self.spec_noise_lin is None:
                self.spec_noise_log = np.zeros(len(self.spec_log))+self.noise_val #if no noise spectrum is given, put here a noise value

    def log_bin(self, mask=True):
        # prepare the spectrum
        if mask:
            mask = (self.wave_lin > self.lam_range[0]) & (self.wave_lin < self.lam_range[1])
            self.spec_lin = self.spec_lin[mask]
            self.spec_lin[np.isnan(self.spec_lin)] = 0
            self.wave_lin = self.wave_lin[mask]
            
            if self.spec_noise_lin is not None:
                self.spec_noise_lin = self.spec_noise_lin[mask]
        lamRange1 = [self.wave_lin[0], self.wave_lin[-1]]  # not exactly the same as lam_range

        if self.velscale is None:
            #get velscale from the rebinning
            self.spec_log, self.logLam, self.velscale = util.log_rebin(lamRange1, self.spec_lin)
        else:
            #if velscale is given
            self.spec_log, self.logLam, self.velscale = util.log_rebin(lamRange1, self.spec_lin, self.velscale)
        # Normalize spectrum to avoid numerical issues
        self.log_median_value = np.nanmedian(self.spec_log)
        self.spec_log = self.spec_log/self.log_median_value
        if self.spec_noise_lin is None:
            self.spec_noise_log = np.zeros(len(self.spec_log))+self.noise_val
        else:
            self.spec_noise_lin[np.isnan(self.spec_noise_lin)] = 9999
            self.spec_noise_log, logLam_noise = util.log_rebin(lamRange1, self.spec_noise_lin, self.velscale)[0:2]
            # rescale in same way as original spectrum
            self.spec_noise_log = self.spec_noise_log/self.log_median_value
        
        #check if they have the same length
        if not len(self.spec_noise_log) == len(self.spec_log):
            #print('AAAH')
            #add something to the spec_noise_log
            if logLam_noise[-1] < self.logLam[-1]:
                spec_noise_log_new = np.zeros(len(self.spec_log))+np.nanmean(self.spec_noise_log)
                spec_noise_log_new[:-1] = self.spec_noise_log
                self.spec_noise_log = spec_noise_log_new
                

    def plot(self, save=False, titel='Spectrum.png', direct='./'):
        fig, ax = plt.subplots()
        ax.plot(self.wave_lin, self.spec_lin, color='k')
        ax.set_xlim(self.lam_range)
        ax.set_xlabel(r'$\lambda$ [$\AA$]')
        ax.set_ylabel('Flux')

    def set_noise(self, new_noise):
        self.spec_noise_log = new_noise

    def vary_spec(self, bestfit, res):
        # The residual is taken from the ppxf fit, has the same size as the masked spec-log
        np.random.seed()
        self.spec_log = bestfit + res*(np.random.choice([-1, 1], len(self.spec_log)))


def ppxf_wrapper_kinematics(ppw):
    '''
    Fitting for the kinematics
    '''
    
    start = [ppw.vel, ppw.sig]
    ppw.start = start
    if ppw.gas_fit:
        # print('Gas fit!')
        stars_templates = ppw.templates.templates.reshape(ppw.templates.templates.shape[0], -1)


        # Construct a set of Gaussian emission line templates.
        # Estimate the wavelength fitted range in the rest frame.
        #
        lam_range_gal = ppw.Spec.lam_range
        gas_templates, gas_names, line_wave = util.emission_lines(
            ppw.templates.log_lam_temp, lam_range_gal, 2.51)

        forbidden_lines_mask = np.array(["[" in a for a in gas_names])
        # print(np.shape(gas_templates))
        balmer_line_templates = gas_templates[:, ~forbidden_lines_mask]
        all_templates = np.column_stack([stars_templates, gas_templates])
        templates_balmer_only = np.column_stack(
            [stars_templates, balmer_line_templates])  # Not implemented

        n_temps = stars_templates.shape[1]
        n_forbidden = np.sum(["[" in a for a in gas_names])  # forbidden lines contain "[*]"
        n_balmer = len(gas_names) - n_forbidden

        # Assign component=0 to the stellar templates, component=1 to the Balmer
        # gas emission lines templates and component=2 to the forbidden lines.
        component = [0]*n_temps + [1]*n_balmer + [2]*n_forbidden
        gas_component = np.array(component) > 0  # gas_component=True for gas templates
        component_balmer_only = [0]*n_temps + [1]*n_balmer

        # Fit (V, sig, h3, h4) moments=4 for the stars
        # and (V, sig) moments=2 for the two gas kinematic components
        moments = [ppw.moments, 2, 2]
        moments_balmer_only = [moments, 2]

        # Adopt the same starting value for the stars and the two gas components
        start = [[ppw.vel, ppw.sig], [ppw.vel, ppw.sig], [ppw.vel, ppw.sig]]
        start_balmer_only = [[ppw.vel, ppw.sig], [ppw.vel, ppw.sig]]
        '''
        gas_names = gas_names[~forbidden_lines_mask]
        templates = templates_balmer_only
        component = component_balmer_only
        gas_component = np.array(component) > 0
        moments = np.array(moments_balmer_only)
        start = start_balmer_only
        '''

        all_templates = all_templates
        component = component
        gas_component = np.array(component) > 0
        moments = np.array(moments)
        ppw.start = start

    else:
        all_templates = ppw.templates.templates
        gas_names = None
        component = 0
        gas_component = None
        start = [ppw.vel, ppw.sig]
        moments = ppw.moments

    lam = np.exp(ppw.templates.log_lam_temp)
    lam_range_temp = [lam.min(), lam.max()]
    dv = 299792.458*(ppw.templates.log_lam_temp[0] - ppw.Spec.logLam[0])
    go = True
    count = 0
    while go:
        # Fitting several times in case the initial guess was off
        z = np.exp(ppw.vel/299792.458) - 1
        # start = [vel, sig]

        goodPixels = sup.determine_goodpixels(
            ppw.Spec.logLam, lam_range_temp, z, mask_file=ppw.mask_file)
        #print(len(goodPixels))
        #print(start)
        pp = ppxf.ppxf(all_templates, ppw.Spec.spec_log, ppw.Spec.spec_noise_log, ppw.Spec.velscale, ppw.start,
                       goodpixels=goodPixels, plot=False, moments=moments, quiet=True,
                       degree=ppw.degree, mdegree=0,
                       component=component, gas_component=gas_component,
                       gas_names=gas_names, gas_reddening=None, lam=np.exp(ppw.Spec.logLam), 
                       lam_temp=np.exp(ppw.templates.log_lam_temp), velscale_ratio = ppw.Spec.velscale_ratio)
        count += 1
        vel_best_fit = pp.sol[0]
        if ppw.gas_fit:
            vel_best_fit = pp.sol[0][0]
        if abs(ppw.vel - vel_best_fit) < 0.5 or count >= 5:
            go = False
        else:
            ppw.vel = vel_best_fit
    return pp


def ppxf_wrapper_stellar_pops(ppw, regul=None):
    """
    # only doing the population fit with fixed kinematics
    """
    if ppw.pp is None:
        print('Do kinematic fit first! and supply pp')
    else:
        ppw.start = ppw.pp.sol
    
    if not ppw.gas_fit:
        if len(ppw.start) > 0:
            start = ppw.start
        else:
            start = [ppw.vel, ppw.sig]
            ppw.start = start

    lam = np.exp(ppw.templates.log_lam_temp)
    lam_range_temp = [lam.min(), lam.max()]
    dv = 299792.458*(ppw.templates.log_lam_temp[0] - ppw.Spec.logLam[0]) #not needed anymore

    if ppw.gas_fit:
        if not ppw.quiet:
            print('Gas fit!')
        stars_templates = ppw.templates.templates.reshape(ppw.templates.templates.shape[0], -1)
        if len(ppw.start) <= 0:
            vel_stars = ppw.vel[0]
            sig_stars = ppw.vel[1]
            vel_gas = ppw.sig[0]
            sig_gas = ppw.sig[1]
        else:
            vel_stars = ppw.start[0][0]
            sig_stars = ppw.start[0][1]
            vel_gas = ppw.start[1][0]
            sig_gas = ppw.start[1][1]
            
        # Construct a set of Gaussian emission line templates.
        # Estimate the wavelength fitted range in the rest frame.
        #
        lam_range_gal = ppw.Spec.lam_range
        gas_templates, gas_names, line_wave = util.emission_lines(
            ppw.templates.log_lam_temp, lam_range_gal, 2.51)

        forbidden_lines_mask = np.array(["[" in a for a in gas_names])
        # print(np.shape(gas_templates))
        balmer_line_templates = gas_templates[:, ~forbidden_lines_mask]
        all_templates = np.column_stack([stars_templates, gas_templates])
        templates_balmer_only = np.column_stack([stars_templates, balmer_line_templates])

        n_temps = stars_templates.shape[1]
        n_forbidden = np.sum(["[" in a for a in gas_names])  # forbidden lines contain "[*]"
        n_balmer = len(gas_names) - n_forbidden

        # Assign component=0 to the stellar templates, component=1 to the Balmer
        # gas emission lines templates and component=2 to the forbidden lines.
        component = [0]*n_temps + [1]*n_balmer + [2]*n_forbidden
        gas_component = np.array(component) > 0  # gas_component=True for gas templates
        component_balmer_only = [0]*n_temps + [1]*n_balmer

        # Adopt the same starting value for the stars and the two gas components
        # start = [[vel, sig], [vel, sig], [vel, sig]]
        start_balmer_only = [[vel_stars, sig_stars], [vel_gas, sig_gas]]
        '''
        # only balmer lines
        gas_names = gas_names[~forbidden_lines_mask]
        templates = templates_balmer_only
        component = component_balmer_only
        gas_component = np.array(component) > 0

        start = start_balmer_only
        '''

        gas_names = gas_names
        all_templates = all_templates  # _balmer_only
        component = component  # _balmer_only
        gas_component = np.array(component) > 0

        if len(ppw.start) > 0:
            #print(ppw.moments)
            ppw.start = ppw.start
            #moments = [ppw.moments, 2, 2]
            #moments = np.array(moments)
            #ppw.moments = moments
        else:
            start = [[vel_stars, sig_stars], [vel_gas, sig_gas], [vel_gas, sig_gas]]
            ppw.start = start
        z = np.exp(vel_stars/299792.458) - 1

    else:
        all_templates = ppw.templates.templates
        gas_names = None
        component = 0
        gas_component = None
        z = np.exp(ppw.vel/299792.458) - 1
      # Relation between velocity and redshift in pPXF

    goodPixels = sup.determine_goodpixels(
            ppw.Spec.logLam, lam_range_temp, z, mask_file=ppw.mask_file)

    reg_dim = all_templates.shape[1:]
    
    if regul is None:
        #to set regul manually in the function call
        regul = ppw.regul
    pp = ppxf.ppxf(all_templates, ppw.Spec.spec_log, ppw.Spec.spec_noise_log, ppw.Spec.velscale, ppw.start, lam=np.exp(ppw.Spec.logLam),
                   goodpixels=goodPixels, plot=False, moments=-ppw.moments, degree=-1, quiet=ppw.quiet, lam_temp=np.exp(ppw.templates.log_lam_temp),
                   clean=False, mdegree=ppw.mdegree, regul=regul, reg_dim=reg_dim,
                   component=component, gas_component=gas_component,
                   gas_names=gas_names, gas_reddening=None, velscale_ratio = ppw.Spec.velscale_ratio)

    return pp


def ppxf_wrapper(wave, spec_lin, noise_spec=None, fwhm=-1, galaxy='FCC47', vel=1366, sig=127, plot=True, kin_only=True,
              moments=2, degree=12, mdegree=8, regul=0, quiet=True,  lam_range=[4700, 2.4e4], plot_kin_title='Spectrum_kin_fit.png', 
              plot_pop_title='Spectrum_pop_fit.png',  plot_out='./', save_plots=False, templates_path=None, 
              age_lim=None, metal_lim=None,  abun_fit=False, abun_prefix='alpha', mask_file=None, 
              logbin=True, velscale=0, gas_fit=False, no_kin_fit=False, start=[], light_weighted=False, templates=None,
              instrument = 'MUSE', ssp_models = 'EMILES', velscale_ratio=1, pp=None):
    """
    The main function of ppxf_MUSE

    Runs ppxf on a given spectrum (linearly binned)

    Args:
        spec_lin (array): mandatory, the spectrum
        wave (array): mandatory, the wavelength array
        fwhm (float): LSF FWHM of the input spectrum. Default is 2.8. If set
                      to -1, the Gueron formula is used
        galaxy (str): name of the galaxy. Mainly used for the plots
        vel (float): initial guess of the line-of-sight velocity
        sig (float): initial guess of the velocity dispersion
        plot (bool): If set to true, plots will be created
        kin_only (bool): If set to true, only kinematics will be fitted
        moments (int): moments of the LOSVD that will be fitted, default is 2
        degree (int): degree of additive polynomials, used for the kinematic fit
        mdegree (int): degree of the multiplicative polynomials used for the pop fit
        regul (float): regularisation parameters. Significantly slows down the fit
        quiet (bool): if set to true, no output will be printed
        lam_range (tuple): lower and higher limit of the fitted wavelengths

    Returns:
        pp_out: ppxf object of the result (default)
        if return_pp is false, an array of the output (vel, sig etc) will be returned

    """
    dir = path.dirname(path.realpath(__file__)) + '/templates/'
    
    #first initialize the ppw object
    ppw = ppwrapper(wave=wave, spec_lin=spec_lin, noise_spec=noise_spec, fwhm=fwhm, galaxy=galaxy, vel=vel, sig=sig, 
                    plot=plot, kin_only=kin_only, moments=moments, degree=degree, mdegree=mdegree, regul=regul,
                    quiet=quiet,  lam_range=lam_range, plot_kin_title=plot_kin_title, 
              plot_pop_title=plot_pop_title,  plot_out=plot_out, save_plots=save_plots, templates_path=templates_path, 
              age_lim=age_lim, metal_lim=metal_lim, abun_fit=abun_fit, mask_file=mask_file, 
              logbin=logbin, velscale=velscale, gas_fit=gas_fit, no_kin_fit=no_kin_fit, start=start, light_weighted=light_weighted,
              templates=templates, instrument = instrument, ssp_models = ssp_models, velscale_ratio=velscale_ratio, pp=pp)
    
    #get the Spectrum object and the templates
    ppw.initialize_spectrum_and_templates(dir=dir)
                    
    # First do kinematics
    if not ppw.no_kin_fit:
        pp_kin = ppxf_wrapper_kinematics(ppw)
        ppw.pp = pp_kin
        if ppw.gas_fit:
            ppw.vel = pp_kin.sol[0][0]
            ppw.sig = pp_kin.sol[0][1]
            if moments == 4:
                ppw.h3 = pp_kin.sol[0][2]
                ppw.h4 = pp_kin.sol[0][3]
            #pp_kin.gas_flux = pp_kin.gas_flux * Spec.log_median_value
        else:
            ppw.vel = pp_kin.sol[0]
            ppw.sig = pp_kin.sol[1]
            if ppw.moments == 4:
                ppw.h3 = pp_kin.sol[2]
                ppw.h4 = pp_kin.sol[3]
        if not ppw.quiet:
            print(
                'V = {0} km/s, sig = {1} km/s'.format(np.round(vel, 2), np.round(sig, 2)))
        if ppw.plot:
            ppw.plot_kin_fit()
            #plotting.plot_pp_kin(pp_kin, title=ppw.plot_kin_title, direct=ppw.plot_out,
             #                    save=ppw.save_plots, gas_fit=ppw.gas_fit, instrument=ppw.instrument)
    if ppw.kin_only:
        return ppw
    
        
    # Then to population fit
    else:
        if ppw.regul == 0:
            pp_pop = ppxf_wrapper_stellar_pops(ppw)
        else:
            # If regularisation is not 0, first have to rescale the noise vector
            pp_unregul = ppxf_wrapper_stellar_pops(ppw, regul=0)
            # rescale the noise vector to get a chi2 = 1 in the unregularized fit
            ppw.Spec.set_noise(ppw.Spec.spec_noise_log * np.sqrt(pp_unregul.chi2))

            pp_pop = ppxf_wrapper_stellar_pops(ppw)
        
        ppw.pp = pp_pop
        
        if abun_fit:
            #not implemented yet!
            age, metal, abun = sup.get_age_metal_abun(pp_pop, ppw.templates, quiet=quiet)
            
            ppw.age = age
            ppw.metal = metal
            ppw.abun = abun
            if plot:
                plotting.plot_pp_pops_abun(pp_pop, ppw.templates, title=plot_pop_title,
                                           direct=plot_out, save=save_plots, abun_prefix=abun_prefix)
                plotting.plot_pp_weights_abun(pp_pop, ppw.templates, title='Weights.png',
                                              direct=plot_out, save=save_plots, abun_prefix=abun_prefix)
                plotting.plot_pp_pops_abun_full_weights(pp_pop, ppw.templates, title='Fit_with_weights.png',
                                                        direct=plot_out, save=save_plots, abun_prefix=abun_prefix)
        else:
            
            ppw.get_age_metal()
            if plot:
                ppw.plot_pop_fit()

        
    return ppw


def ppxf_wrapper_test_regul(wave, spec_lin, noise_spec=None, fwhm=2.8, spec_noise=None, galaxy='FCC47', vel=1366, sig=127, plot=False,
                         moments=2, degree=20, mdegree=20, regul=70, quiet=True, plot_pop_title='Spectrum_pop_fit.png',
                         plot_out='./', save_plots=False, lam_range=[3540, 9000], templates_path=None, logbin=True,
                         abun_fit=False, abun_prefix='alpha', age_lim=False, metal_lim=None, mask_file=None, gas_fit=False, 
                         light_weighted=False, instrument='MUSE', ssp_models='EMILES', templates=None, velscale_ratio=1, 
                         kin_only=False):
    """
    Testing regul parameters
    """
    if kin_only:
        print('regul only works with pop fitting')
        return
        
    dir = path.dirname(path.realpath(__file__)) + '/templates/'
    
    #first initialize the ppw object
    ppw = ppwrapper(wave=wave, spec_lin=spec_lin, noise_spec=noise_spec, fwhm=fwhm, galaxy=galaxy, vel=vel, sig=sig, 
                    plot=plot, moments=moments, degree=degree, mdegree=mdegree, regul=regul,
                    quiet=quiet,  lam_range=lam_range, 
              plot_pop_title=plot_pop_title,  plot_out=plot_out, save_plots=save_plots, templates_path=templates_path, 
              age_lim=age_lim, metal_lim=metal_lim, abun_fit=abun_fit, mask_file=mask_file, 
              logbin=logbin, gas_fit=gas_fit, light_weighted=light_weighted,
              templates=templates, instrument = instrument, ssp_models = ssp_models, velscale_ratio=velscale_ratio)
              
    ppw.initialize_spectrum_and_templates(dir=dir)
        
    # First do kinematics
    pp_kin = ppxf_wrapper_kinematics(ppw)
    ppw.pp = pp_kin
    if not ppw.quiet:
        print(
            'V = {0} km/s, sig = {1} km/s'.format(np.round(pp_kin.sol[0], 2), np.round(pp_kin.sol[1], 2)))

    print('Doing the unregularized fit!')
    pp_unregul = ppxf_wrapper_stellar_pops(ppw, regul=0)
    ppw.pp = pp_unregul
    ppw.plot_pop_fit()
        
    # rescale the noise vector to get a chi2 = 1 in the unregularized fit
    ppw.Spec.set_noise(ppw.Spec.spec_noise_log * np.sqrt(pp_unregul.chi2))
    for regul_i in ppw.regul:
        print('Doing the regularized fit with regul = {0}!'.format(regul_i))
        pp_regul = ppxf_wrapper_stellar_pops(ppw, regul=regul_i)  # do the regularized fit
        #plot
        ppw.pp = pp_regul
        ppw.plot_pop_fit()

        desired = np.sqrt(2*pp_regul.goodpixels.size)
        current = (pp_regul.chi2 - 1)*pp_regul.goodpixels.size
        print('Desired: {0}'.format(np.round(desired, 2)))
        print('Current: {0}'.format(np.round(current, 2)))
        
    return ppw

def the_funct(i, ppw):
    # Randomize the spectrum based on the residual

    # First do kinematics
    res = ppw.pp.galaxy - ppw.pp.bestfit
    ppw.Spec.vary_spec(ppw.pp.bestfit, res)
    
    pp_kin = ppxf_wrapper_kinematics(ppw)
    ppw.pp = pp_kin
    if not ppw.quiet:
        print(
            'V = {0} km/s, sig = {1} km/s'.format(np.round(pp_kin.sol[0], 2), np.round(pp_kin.sol[1], 2)))

    if not ppw.gas_fit:
        if ppw.moments == 2:
            start = [pp_kin.sol[0], pp_kin.sol[1]]
        if ppw.moments == 4:
            start = [pp_kin.sol[0], pp_kin.sol[1], pp_kin.sol[2], pp_kin.sol[3]]
    else:
        if ppw.moments == 2:
            start = [pp_kin.sol[0][0], pp_kin.sol[0][1],
                     pp_kin.sol[1][0], pp_kin.sol[1][1]]  # stars and gas
            #start = [pp_kin.sol[0], pp_kin.sol[1], pp_kin.sol[2]]
        if ppw.moments == 4:
            #start = [pp_kin.sol[0], pp_kin.sol[1], pp_kin.sol[2]]
            start = [pp_kin.sol[0][0], pp_kin.sol[0][1], pp_kin.sol[0]
                     [2], pp_kin.sol[0][3], pp_kin.sol[1][0], pp_kin.sol[1][1]]
 
    if ppw.kin_only:
        result = start
        ppw.pp = pp_kin
    # Then to population fit
    else:
        if ppw.gas_fit:
            vel = pp_kin.sol[0][0]
            sig = pp_kin.sol[0][1]
            #print(vel, sig)
        else:
            vel = pp_kin.sol[0]
            sig = pp_kin.sol[1]
            
        pp_pop = ppxf_wrapper_stellar_pops(ppw)
        ppw.pp = pp_pop
        
        if not ppw.gas_fit:
            if ppw.abun_fit:
                age, ppw.metal, alpha = sup.get_age_metal_abun(pp_pop, ppw.templates, quiet=ppw.quiet)
                if ppw.moments == 2:
                    result = [pp_pop.sol[0], pp_pop.sol[1], age, metal, alpha]
                if ppw.moments == 4:
                    result = [pp_pop.sol[0], pp_pop.sol[1],
                              pp_pop.sol[2], pp_pop.sol[3], age, metal, alpha]
            else:
                ppw.get_age_metal()
                age = ppw.age
                metal = ppw.metal
                if ppw.moments == 2:
                    result = [pp_pop.sol[0], pp_pop.sol[1], age, metal]
                if ppw.moments == 4:
                    result = [pp_pop.sol[0], pp_pop.sol[1],
                              pp_pop.sol[2], pp_pop.sol[3], age, metal]
        else:
            if ppw.abun_fit:
                age, metal, alpha = sup.get_age_metal_abun(pp_pop, ppw.templates, quiet=ppw.quiet)
                if ppw.moments == 2:
                    result = [pp_pop.sol[0][0], pp_pop.sol[0][1],
                              pp_pop.sol[1][0], pp_pop.sol[1][1], age, metal, alpha]
                if ppw.moments == 4:
                    result = [pp_pop.sol[0][0], pp_pop.sol[0][1], pp_pop.sol[0][2], pp_pop.sol[0][3], pp_pop.sol[1][0], pp_pop.sol[1][1],
                              age, metal, alpha]
            else:
                ppw.get_age_metal()
                age = ppw.age
                metal = ppw.metal
                if ppw.moments == 2:
                    result = [pp_pop.sol[0][0], pp_pop.sol[0][1],
                              pp_pop.sol[1][0], pp_pop.sol[1][1], age, metal]
                if ppw.moments == 4:
                    result = [pp_pop.sol[0][0], pp_pop.sol[0][1], pp_pop.sol[0][2],
                              pp_pop.sol[0][3], pp_pop.sol[1][0], pp_pop.sol[1][1], age, metal]
        #ppw.pp = pp_pop
    return result

def ppxf_wrapper_MC(wave, spec_lin, noise_spec=None, fwhm=2.8,  kin_only=False, galaxy='FCC47', vel=1366, sig=127, 
                 moments=2, degree=12, mdegree=8, regul=0, quiet=True,  lam_range=[3540, 8900], templates_path=None,
                 n=300, cores=4, savetxt=True, save_plots=True, filebase_MC='Spec_MC', out_dir='./', plot_hist=True, 
                 age_lim=12, metal_lim=None, abun_fit=False, mask_file=None, ssp_models='EMILES', templates=None, logbin=True,
                 gas_fit=False, light_weighted=False, instrument='MUSE', velscale_ratio=1):
    """
    Do ppxf fit of a MUSE spectrum with MC simulations to determine the uncertainties.
    Saves the runs to a file
    """
    dir = path.dirname(path.realpath(__file__)) + '/templates/'
    
    #first initialize the ppw object
    ppw = ppwrapper(wave=wave, spec_lin=spec_lin, noise_spec=noise_spec, fwhm=fwhm, galaxy=galaxy, vel=vel, sig=sig, 
                    kin_only=kin_only, moments=moments, degree=degree, mdegree=mdegree, regul=regul,
                    quiet=quiet,  lam_range=lam_range, templates_path=templates_path, 
                    age_lim=age_lim, metal_lim=metal_lim, abun_fit=abun_fit, mask_file=mask_file, n=n, cores=cores,
                    logbin=logbin, gas_fit=gas_fit, light_weighted=light_weighted, filebase_MC=filebase_MC, out_dir = out_dir,
                    templates=templates, instrument = instrument, ssp_models = ssp_models, velscale_ratio=velscale_ratio, 
                    plot_hist=plot_hist, savetxt=savetxt, save_plots=save_plots)
    
    ppw.initialize_spectrum_and_templates(dir=dir)

    pp_kin = ppxf_wrapper_kinematics(ppw)
    ppw.pp = pp_kin
    
    if ppw.save_plots:
        ppw.plot_kin_fit()
        
    if not quiet:
        print('First fit gave:')
        print(ppw.pp.sol[0], ppw.pp.sol[1])

    partial_func = partial(the_funct, ppw=ppw)

    i = np.arange(ppw.n)
    print(
        'First fit done. Starting the MC fits on {0} cores... this can take a while'.format(ppw.cores))
    pool = mp.Pool(processes=ppw.cores)
    result = pool.map(partial_func, i)
    pool.close()
    pool.join()

    result = np.asarray(result)
    if not ppw.gas_fit:
        v = result[:, 0]
        sig = result[:, 1]
        if ppw.moments == 4:
            i = 2
            h3 = result[:, 2]
            h4 = result[:, 3]
            ppw.filebase_MC += '_mom4'
        else:
            i = 0
        if not ppw.kin_only:
            age = result[:, 2+i]
            metal = result[:, 3+i]
            if abun_fit:
                abun = result[:, 4+i]
        if ppw.plot_hist:
            plotting.plot_hists(result, out_dir=ppw.out_dir, kin_only=ppw.kin_only)

        ppw.v_mc, ppw.dv_mc = np.round(np.nanmean(v), 2), np.round(np.nanstd(v), 2)
        ppw.sig_mc, ppw.dsig_mc = np.round(np.nanmean(sig), 2), np.round(np.nanstd(sig), 2)
        
        if not ppw.quiet:
            print('V = {0} +- {1} km/s'.format(ppw.v_mc, ppw.dv_mc))
            print('sig = {0} +- {1} km/s'.format(ppw.sig_mc, ppw.dsig_mc))
        if ppw.moments == 4:
            ppw.h3_mc, ppw.dh3_mc = np.round(np.nanmean(h3), 2), np.round(np.nanstd(h3), 2)
            ppw.h4_mc, ppw.dh4_mc = np.round(np.nanmean(h4), 2), np.round(np.nanstd(h4), 2)
            if not ppw.quiet:
                print('h3 = {0} +- {1} km/s'.format(ppw.h3_mc, ppw.dh3_mc))
                print('h4 = {0} +- {1} km/s'.format(ppw.h4_mc, ppw.dh4_mc))
        if not ppw.kin_only:
            ppw.age_mc, ppw.dage_mc = np.round(np.nanmean(age), 2), np.round(np.nanstd(age), 2)
            ppw.metal_mc, ppw.dmetal_mc = np.round(np.nanmean(metal), 2), np.round(np.nanstd(metal), 2)
            if not ppw.quiet:
                print('age = {0} +- {1} Gyr'.format(ppw.age_mc, ppw.dage_mc))
                print('[M/H] = {0} +- {1} dex'.format(ppw.metal_mc, ppw.dmetal_mc))
            if ppw.abun_fit:
                ppw.abun_mc, ppw.dabun_mc = np.round(np.nanmean(abun), 2), np.round(np.nanstd(abun), 2)
                if not ppw.quiet:
                    print('Abun = {0} +- {1} dex'.format(ppw.abun_mc, ppw.dabun_mc))
        if ppw.savetxt:
            if ppw.kin_only:
                filename = ppw.out_dir + ppw.filebase_MC + '_{0}_runs_kin_only.dat'.format(ppw.n)
            else:
                filename = ppw.out_dir + ppw.filebase_MC + '_{0}_runs.dat'.format(ppw.n)
            result_list = [v, sig]
            if ppw.moments == 4:
                result_list.append(h3)
                result_list.append(h4)
            if not ppw.kin_only:
                result_list.append(age)
                result_list.append(metal)
                if abun_fit:
                    result_list.append(abun)
            print('Saving {}'.format(filename))
            np.savetxt(filename, np.transpose(result_list), fmt='%.3f')
    else:
        v_stars = result[:, 0]
        sig_stars = result[:, 1]
        if ppw.moments == 4:
            i = 2
            h3 = result[:, 2]
            h4 = result[:, 3]
        else:
            i = 0
        v_gas = result[:, 2+i]
        sig_gas = result[:, 3+i]
        if not ppw.kin_only:
            age = result[:, 4+i]
            metal = result[:, 5+i]
            if ppw.abun_fit:
                abun = result[:, 6+i]
        
        ppw.v_mc, ppw.dv_mc = np.round(np.nanmean(v_stars), 2), np.round(np.nanstd(v_stars), 2)
        ppw.sig_mc, ppw.dsig_mc = np.round(np.nanmean(sig_stars), 2), np.round(np.nanstd(sig_stars), 2)
        
        ppw.v_gas_mc, ppw.dv_gas_mc = np.round(np.nanmean(v_gas), 2), np.round(np.nanstd(v_gas), 2)
        ppw.sig_gas_mc, ppw.dsig_gas_mc = np.round(np.nanmean(sig_gas), 2), np.round(np.nanstd(sig_gas), 2) 
        
        if not ppw.quiet:
            print('V_stars = {0} +- {1} km/s'.format(ppw.v_mc, ppw.dv_mc))
            print('sig_stars = {0} +- {1} km/s'.format(ppw.sig_mc, ppw.dsig_mc))
            print('V_gas = {0} +- {1} km/s'.format(ppw.v_gas_mc, ppw.dv_gas_mc))
            print('sig_gas = {0} +- {1} km/s'.format(ppw.sig_gas_mc, ppw.dsig_gas_mc))
        
        if ppw.moments == 4:
            ppw.h3_mc, ppw.dh3_mc = np.round(np.nanmean(h3), 2), np.round(np.nanstd(h3), 2)
            ppw.h4_mc, ppw.dh4_mc = np.round(np.nanmean(h4), 2), np.round(np.nanstd(h4), 2)
            if not ppw.quiet:
                print('h3 = {0} +- {1} km/s'.format(ppw.h3_mc, ppw.dh3_mc))
                print('h4 = {0} +- {1} km/s'.format(ppw.h4_mc, ppw.dh4_mc))
                
        if not ppw.kin_only:
            ppw.age_mc, ppw.dage_mc = np.round(np.nanmean(age), 2), np.round(np.nanstd(age), 2)
            ppw.metal_mc, ppw.dmetal_mc = np.round(np.nanmean(metal), 2), np.round(np.nanstd(metal), 2)
            if not ppw.quiet:
                print('age = {0} +- {1} Gyr'.format(ppw.age_mc, ppw.dage_mc))
                print('[M/H] = {0} +- {1} dex'.format(ppw.metal_mc, ppw.dmetal_mc))
            if ppw.abun_fit:
                ppw.abun_mc, ppw.dabun_mc = np.round(np.nanmean(abun), 2), np.round(np.nanstd(abun), 2)
                if not ppw.quiet:
                    print('Abun = {0} +- {1} dex'.format(ppw.abun_mc, ppw.dabun_mc))
                
        if ppw.savetxt:
            if ppw.kin_only:
                filename = ppw.out_dir + ppw.filebase_MC + '_{0}_runs_kin_only_w_gas.dat'.format(ppw.n)
            else:
                filename = ppw.out_dir + ppw.filebase_MC + '_{0}_runs_w_gas.dat'.format(ppw.n)
            result_list = [v_stars, sig_stars, v_gas, sig_gas]
            if ppw.moments == 4:
                result_list.append(h3)
                result_list.append(h4)
            if not ppw.kin_only:
                result_list.append(age)
                result_list.append(metal)
                if ppw.abun_fit:
                    result_list.append(abun)
            print('Saving {}'.format(filename))
            np.savetxt(filename, np.transpose(result_list), fmt='%.3f')
    return ppw


if __name__ == "__main__":
    file = '/Users/kfahrion/Documents/PhD/UCD_paper/Other/pPXF/FCC47_UCD_spec.dat'
    wave, spec, spec_bg = np.loadtxt(file, unpack=1)
    wave, var_spec, var_spec_bg = np.loadtxt(
        '/Users/kfahrion/Documents/PhD/UCD_paper/Other/pPXF/FCC47_UCD_var_spec.dat', unpack=1)

    # Path + common name for the EMILES templates
    templates_path = '/Users/kfahrion/Documents/Data/MILES/EMILES_Basti/Ebi1.30*.fits'

    spec = spec - spec_bg
    var_spec = var_spec + var_spec_bg
    ppxf_wrapper_MC(spec, wave, quiet=False, kin_only=True, fwhm=-
                 1, n=5, templates_path=templates_path, galaxy='FCC47')

    # ppxf_MUSE(spec, wave, quiet = False, kin_only = True, fwhm = -1, plot= True)
