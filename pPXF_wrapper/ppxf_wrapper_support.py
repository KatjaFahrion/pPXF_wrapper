from __future__ import print_function

import glob
from os import path
import matplotlib.pylab as plt

import numpy as np
import pPXF_MUSE.miles_support as lib

from mpl_toolkits.axes_grid1 import make_axes_locatable


def read_mask_file(mask_file):
    f = np.loadtxt(mask_file, unpack=1)
    line_starts = f[0]
    line_ends = f[1]
    mask_array = np.zeros((len(line_starts), 2))
    for i in range(len(mask_array)):
        mask_array[i, :] = [line_starts[i], line_ends[i]]
    return mask_array


# %%
def determine_goodpixels(logLam, lamRangeTemp, z=0, AO=True, sky=True, em_lines=False, gal='FCC170', mask_file=None):
    """
    Generates a list of goodpixels to mask a given set of gas emission
    lines. This is meant to be used as input for PPXF.

    :param logLam: Natural logarithm np.log(wave) of the wavelength in
        Angstrom of each pixel of the log rebinned *galaxy* spectrum.
    :param lamRangeTemp: Two elements vectors [lamMin2, lamMax2] with the minimum
        and maximum wavelength in Angstrom in the stellar *template* used in PPXF.
    :param z: Estimate of the galaxy redshift.
    :return: vector of goodPixels to be used as input for pPXF


    # Taken FROM PPXF_UTIL AND ADAPTED
    """
#                     -----[OII]-----    Hdelta   Hgamma   Hbeta   -----[OIII]-----   [OI]    -----[NII]-----   Halpha   -----[SII]-----
    lines = np.array([3726.03, 3728.82, 4101.76, 4340.47, 4861.33, 4958.92,
                      5006.84, 6300.30, 6548.03, 6583.41, 6562.80, 6716.47, 6730.85])

    if mask_file is None:
        mask = np.array([[7586, 7750],
                         [6451, 6495],
                         [6885, 6965],
                         [8100, 8345],
                         [7158, 7211],
                         [7226, 7330],
                         [6283, 6318],
                         [5570, 5582],
                         ])
    else:
        mask = read_mask_file(mask_file)

    dv = lines*0 + 800  # width/2 of masked gas emission region in km/s
    c = 299792.458  # speed of light in km/s

    flag = np.zeros_like(logLam, dtype=bool)
    if em_lines:
        for line, dvj in zip(lines, dv):
            flag |= (np.exp(logLam) > line*(1 + z)*(1 - dvj/c)) \
                & (np.exp(logLam) < line*(1 + z)*(1 + dvj/c))

    flag |= np.exp(logLam) > lamRangeTemp[1]*(1 + z)*(1 - 900/c)   # Mask edges of
    flag |= np.exp(logLam) < lamRangeTemp[0]*(1 + z)*(1 + 900/c)   # stellar library

    skyregions = mask
    if sky:
        for sky_reg in skyregions:
            flag |= (np.exp(logLam) < sky_reg[1]) \
                & (np.exp(logLam) > sky_reg[0])
    AO_region = np.array([5801, 5967])

    if AO:
        #print('Masking AO')
        flag |= (np.exp(logLam) < AO_region[1]) \
            & (np.exp(logLam) > AO_region[0])

    return np.where(flag == 0)[0]


def determine_goodpixels_CaT(logLam, lamRangeTemp, z, AO=False, sky=False, em_lines=True, gal='FCC47'):
    """
    Generates a list of goodpixels to mask a given set of gas emission
    lines. This is meant to be used as input for PPXF.

    :param logLam: Natural logarithm np.log(wave) of the wavelength in
        Angstrom of each pixel of the log rebinned *galaxy* spectrum.
    :param lamRangeTemp: Two elements vectors [lamMin2, lamMax2] with the minimum
        and maximum wavelength in Angstrom in the stellar *template* used in PPXF.
    :param z: Estimate of the galaxy redshift.
    :return: vector of goodPixels to be used as input for pPXF


    # Taken FROM PPXF UTIL AND ADAPTED
    """
#                     -----[OII]-----    Hdelta   Hgamma   Hbeta   -----[OIII]-----   [OI]    -----[NII]-----   Halpha   -----[SII]-----
   # lines = np.array([3726.03, 3728.82, 4101.76, 4340.47, 4861.33, 4958.92, 5006.84, 6300.30, 6548.03, 6583.41, 6562.80, 6716.47, 6730.85])
    lines = np.array([3726.03, 3728.82, 6716.47, 6730.85])
    mask = np.array([
        [8250.03, 8559.31],
        [8614.61, 8694.23],
        [8765.41, 10000.]])
    # ]]])

    dv = lines*0 + 800  # width/2 of masked gas emission region in km/s
    c = 299792.458  # speed of light in km/s

    flag = np.zeros_like(logLam, dtype=bool)
    if em_lines:
        for line, dvj in zip(lines, dv):
            flag |= (np.exp(logLam) > line*(1 + z)*(1 - dvj/c)) \
                & (np.exp(logLam) < line*(1 + z)*(1 + dvj/c))

    flag |= np.exp(logLam) > lamRangeTemp[1]*(1 + z)*(1 - 900/c)   # Mask edges of
    flag |= np.exp(logLam) < lamRangeTemp[0]*(1 + z)*(1 + 900/c)   # stellar library

  #  print(flag)

    skyregions = np.array([[8453, 8510]])  # [8553, 8571]])

    # , [7162, 7367], [7003, 7009], [8486, 8490]])#, [7060, 7400]]) # in AA
    if sky:
        for sky_reg in skyregions:
            flag |= (np.exp(logLam) < sky_reg[1]) \
                & (np.exp(logLam) > sky_reg[0])

    AO_region = np.array([5795, 5976])
    if AO:
        flag |= (np.exp(logLam) < AO_region[1]) \
            & (np.exp(logLam) > AO_region[0])

    return np.where(flag == 0)[0]


def get_age_metal(pp, miles, quiet=True):
    """
    Get age & metallicity from pp object for a alpha-fixed fit
    """
    #weights = pp.weights[~gas_component]
    weights = pp.weights[~pp.gas_component].reshape(
        miles.n_ages, miles.n_metal)/pp.weights[~pp.gas_component].sum()
    mean_age, mean_metal = miles.mean_age_metal(weights, quiet=True)

    xgrid = miles.age_grid
    ygrid = miles.metal_grid
    mean_age = np.sum(weights*xgrid)/np.sum(weights)
    mean_metal = np.sum(weights*ygrid)/np.sum(weights)

    if not quiet:
        print('Age = {0} Gyr, [M/H] = {1}'.format(np.round(mean_age, 2), np.round(mean_metal, 2)))
    return mean_age, mean_metal


def get_best_SSP(pp, miles, quiet=False):
    weights = pp.weights[: 636].reshape(miles.n_ages, miles.n_metal)/pp.weights[: 636].sum()
    xgrid = miles.age_grid
    ygrid = miles.metal_grid
    ind = weights == np.max(weights)
    best_age = xgrid[ind][0]
    best_metal = ygrid[ind][0]
    if not quiet:
        print('Age = {0} Gyr, [M/H] = {1}'.format(np.round(best_age, 2), np.round(best_metal, 2)))
    return best_age, best_metal


def get_age_metal_with_errors(pp, miles, quiet=True):
    """
    Get age & metallicity and weighted errors from pp object for a alpha-fixed fit
    """
    weights = pp.weights[: 636].reshape(miles.n_ages, miles.n_metal)/pp.weights[: 636].sum()
    mean_age, mean_metal = miles.mean_age_metal(weights, quiet=True)

    xgrid = miles.age_grid
    ygrid = miles.metal_grid
    mean_age = np.sum(weights*xgrid)/np.sum(weights)
    mean_metal = np.sum(weights*ygrid)/np.sum(weights)
    std_age = np.sqrt(np.sum(weights)/(np.sum(weights)**2 - np.sum(weights**2))
                      * np.sum(weights * (xgrid - mean_age)**2))
    std_metal = np.sqrt(np.sum(weights)/(np.sum(weights)**2 - np.sum(weights**2))
                        * np.sum(weights * (ygrid - mean_metal)**2))

    if not quiet:
        print('Age = {0} Gyr, [M/H] = {1}'.format(np.round(mean_age, 2), np.round(mean_metal, 2)))
    return mean_age, std_age, mean_metal, std_metal


def get_ML(pp, miles, plot_grid=False):
    weights = pp.weights[: 636].reshape(miles.n_ages, miles.n_metal)/pp.weights[: 636].sum()

    xgrid = miles.age_grid
    ygrid = miles.metal_grid

    mass_grid = np.empty_like(weights)
    lum_grid = np.empty_like(weights)
    for j in range(miles.n_ages):
        for k in range(miles.n_metal):
          #  print(ygrid[j, k], xgrid[j, k])
            mass, lum = get_ml_i(metal=ygrid[j, k], age=xgrid[j, k])
            mass_grid[j, k] = mass
            lum_grid[j, k] = lum

    mean_mass = np.sum(weights*mass_grid)/np.sum(weights)
    mean_lum = np.sum(weights*lum_grid)/np.sum(weights)

    x = xgrid[:, 0]
    y = ygrid[0, :]
    xb = (x[1:] + x[: -1])/2  # grid borders
    yb = (y[1:] + y[: -1])/2
    xb = np.hstack([1.5*x[0] - x[1]/2, xb, 1.5*x[-1] - x[-2]/2])
    yb = np.hstack([1.5*y[0] - y[1]/2, yb, 1.5*y[-1] - y[-2]/2])

    if plot_grid:
        mls = mass_grid/lum_grid
        xlabel = "Age"
        ylabel = "[M/H]"
        fig, ax1 = plt.subplots()
        pc = ax1.pcolormesh(xb, yb, mls.T, cmap='gist_heat_r', edgecolors='lightgray', lw=0.01)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.set_ylim(min(yb), max(yb))
        ax1.set_xlim(min(xb), max(xb))
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", "1.5%", pad="1.5%")
        cbar = plt.colorbar(pc, cax=cax)
        cbar.set_label(r'M/L')

    mlpop = np.sum(weights*mass_grid)/np.sum(weights*lum_grid)
    err_mass = np.sqrt(np.sum(weights)/(np.sum(weights)**2 - np.sum(weights**2))
                       * np.sum(weights * (mass_grid - mean_mass)**2))
    err_lum = np.sqrt(np.sum(weights)/(np.sum(weights)**2 - np.sum(weights**2))
                      * np.sum(weights * (lum_grid - mean_lum)**2))
    err_mlpop = np.sqrt((mlpop/mean_mass * err_mass)**2 + (mlpop/mean_lum * err_lum)**2)
    return mlpop, err_mlpop


def get_ml_grid(pp, miles, file='/Users/kfahrion/Documents/Scripts_new/My_packages/pPXF_MUSE/pPXF_MUSE/HST_bi_iTp0.00.dat', index=8,
                imf_index=0, age_index=3, metal_index=2, mass_index=4, m_ref=5.21):
    """
    Returns a grid of mass-to-light ratios for all E-MILES age & metallicity weights
    """
    weights = pp.weights[: 636].reshape(miles.n_ages, miles.n_metal)/pp.weights[: 636].sum()

    xgrid = miles.age_grid
    ygrid = miles.metal_grid

    mass_grid = np.empty_like(weights)
    lum_grid = np.empty_like(weights)
    for j in range(miles.n_ages):
        for k in range(miles.n_metal):
          #  print(ygrid[j, k], xgrid[j, k])
            mass, lum = get_ml_i(metal=ygrid[j, k], age=xgrid[j, k], file=file, index=index,
                                 imf_index=imf_index, age_index=age_index, metal_index=metal_index, mass_index=mass_index, m_ref=m_ref)
            mass_grid[j, k] = mass
            lum_grid[j, k] = lum

    ml_grid = mass_grid/lum_grid

    return ml_grid


def readfile_ML(file='/Users/kfahrion/Documents/Scripts_new/My_packages/pPXF_MUSE/pPXF_MUSE/HST_bi_iTp0.00.dat', index=8,
                imf_index=0, age_index=3, metal_index=2, mass_index=4):
    """
    Read the SSP prediction file
    """
    f = open(file)
    lines = f.readlines()
    mass = []
    mag = []
    age = []
    metal = []
    imf = []
    for line in lines[1:]:
        line = line.split()
        if imf_index == 0:
            imf_i = line[0].split('Z')[0][3:]
            imf.append(float(imf_i))
        else:
            imf.append(float(line[imf_index]))
        age.append(float(line[age_index]))
        metal.append(float(line[metal_index]))
        mass.append(float(line[mass_index]))
        mag.append(float(line[index]))
    return np.array(imf), np.array(age), np.array(metal), np.array(mass), np.array(mag)
# %%


def get_ml_i(metal, age, file='/Users/kfahrion/Documents/Scripts_new/My_packages/pPXF_MUSE/pPXF_MUSE/HST_bi_iTp0.00.dat', index=8,
             imf_index=0, age_index=3, metal_index=2, mass_index=4, m_ref=5.21):
    """
    Extract the mass-to-light ratio of a given metal & age weight
    Return: mass, luminosity (in solar luminosities, using F475W filter)
    """
    imfs, ages, metals, masses, mags = readfile_ML(
        file=file, index=index, imf_index=imf_index, age_index=age_index, metal_index=metal_index, mass_index=mass_index)
    t = np.where(imfs == 1.30)
    imfs, ages, metals, masses, mags = imfs[t], ages[t], metals[t], masses[t], mags[t]
    s = np.where(ages == age)
    imfs, ages, metals, masses, mags = imfs[s], ages[s], metals[s], masses[s], mags[s]
    i = np.where(metals == metal)
    imfs, ages, metals, masses, mags = imfs[i], ages[i], metals[i], masses[i], mags[i]

    lum = 10**(-0.4*(mags[0]-m_ref))

    return masses[0], lum


def get_age_metal_abun(pp, miles, quiet=True):
    """
    Get age, alpha and metallicity from a pp object for alpha-variable fit
    INPUT: pp, miles, quiet = True
    Output: mean_age, mean_metal, mean_alpha
    """
    #weights = pp.weights.reshape(miles.n_ages, miles.n_metal, miles.n_alphas)/pp.weights.sum()
    weights = pp.weights[~pp.gas_component].reshape(
        miles.n_ages, miles.n_metal, miles.n_abuns)/pp.weights[~pp.gas_component].sum()

    xgrid = miles.age_grid
    ygrid = miles.metal_grid
    zgrid = miles.abun_grid
    mean_age = np.sum(weights*xgrid)/np.sum(weights)
    mean_metal = np.sum(weights*ygrid)/np.sum(weights)
    mean_abun = np.sum(weights*zgrid)/np.sum(weights)

    if not quiet:
        print('Age = {0} Gyr, [M/H] = {1}'.format(np.round(mean_age, 2), np.round(mean_metal, 2)))
        print('Abun = {0}'.format(np.round(mean_abun, 2)))
    return mean_age, mean_metal, mean_abun
