# pPXF_wrapper

A wrapper around ppxf (Cappellari & Emsellem 2004, PASP, 116, 138C; Cappellari 2017, MNRAS, 466, 798) to use with all sorts of spectra, at the moment works well for MUSE and SINFONI.


### Prerequisites

Requires python 3.x with astropy and ppxf


## How to install

* Click on "Clone or download" and chose Download ZIP
* Unpack the ZIP archive somewhere and navigate into the folder in the command line
* Type in the comment line:
```
pip install -e .
```
This should install the package to your python path

## How to use
In IPython or any python script, use:
```
import pPXF_wrapper.ppxf_wrapper as ppxf_wrapper
```
The standard function can be used with
```
ppw = ppxf_wrapper.ppxf_wrapper(wave, spec)
```
Where wave is a numpy array containing the wavelength axis (assumed to be in AA for MUSE, and in micron for SINFONI). spec is a numpy array containing the corresponding spectrum.
This will run ppxf on a given spectrum and return a ppxf_wrapper object that contains all the options that went into the fit, the templates and the results:

```
templates = ppw.templates #templates object containing the templates and some info around them
pp = ppw.pp #the standard pp object returned by ppxf, the velocity is in pp.sol[0] for example
Spec = ppw.Spec #Spectrum object (contains linear and log-binned spectrum + noise that was used in the fit)
```

## Important parameters
While the example above might run, there are several key parameters that should be set to make things easier:

```
ppw = ppxf_wrapper.ppxf_wrapper(wave, spec, noise_spec=noise_spec, lam_range=[4700, 7100], vel=1400,         
                                mask_file=mask_file, degree=12, mdegree=8, kin_only=False, save_plots=True, 
                                plot_out='./Plots/', instrument = "MUSE", ssp_models='EMILES")
```
Where:
* noise_spec: the corresponding variance spectrum. Fitting also works without (then a constant value is assumed)
* lam_range: The fitting range (in the same units as the wavelength array)
* vel: velocity guess. This often needs to be set to something close to make the fit work at all
* mask_file: This can point to a line mask file where any sky residual lines can be masked (see the examples). This makes it quick to mask any lines
* degree: degree of additative polynomials for fitting with ppxf (used for kinematic fits)
* mdegree: degree of multiplicative polynomials for fitting with ppxf (used only for stellar pop fits)
* kin_only: If True, only the kinematics are fitted (faster)
* save_plots: If True, plots are saved to the directory given in plot_out
* instrument: can be MUSE or SINFONI. Will change if the wavelength is assumed to be in AA or micron and how the spectral resolution is applied
* ssp_models: The single stellar population models that are used in the fit. Can be 'EMILES' (E-MILES models for MUSE or SINFONI), "XSL" (X-Shooter spectral library SSPs for MUSE or SINFONI), "MILES_solar" (solar-scaled MILES models for MUSE only), "MILES_alpha" (alpha-enhanced models for MUSE only)

## Acknowledgments

* Based on the ppxf source code: https://pypi.org/project/ppxf/ (Cappellari & Emsellem 2004, PASP, 116, 138C; Cappellari 2017, MNRAS, 466, 798)
* MILES models from http://miles.iac.es (Vazdekis et al. 2012, MNRAS, 424, 157; Vazdekis et al. 2015, MNRAS, 449, 1177V; Vazdekis et al. 2016, MNRAS, 463, 3409)
* XSL SSP models from http://xsl.u-strasbg.fr/page_ssp.html (Verro et al. 2022, A&A, 661, 50)


## TO DO

* Implement alpha abundance fitting (in templates mainly + new plotting functions)
* Make jupyter notebook with example
* implement fit function call in ppw object for convenience
* test test test