# pPXF_MUSE

A wrapper around ppxf (Cappellari 2017) to use with all sorts of spectra.
Only for personal use!


### Prerequisites

Requires python 3.x with astropy



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
import pPXF_MUSE
```
The standard function can be used with
```
pPXF_MUSE.ppxf_MUSE(spec, wave)
```
This will run ppxf on a given spectrum


## Acknowledgments

* Based on the ppxf source code. Please do not redistribute
