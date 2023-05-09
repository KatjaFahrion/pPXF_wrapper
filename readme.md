# pPXF_wrapper

A wrapper around ppxf (Cappellari 2017) to use with all sorts of spectra, at the moment works well for MUSE and SINFONI


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
This will run ppxf on a given spectrum and return a ppxf_wrapper object that contains all the options that went into the fit, the templates and the results:

```
templates = ppw.templates #templates object containing the templates and some info around them
pp = ppw.pp #the standard pp object returned by ppxf, the velocity is in pp.sol[0] for example
Spec = ppw.Spec #Spectrum object (contains linear and log-binned spectrum + noise that was used in the fit)
```


## Acknowledgments

* Based on the ppxf source code. Please do not redistribute
