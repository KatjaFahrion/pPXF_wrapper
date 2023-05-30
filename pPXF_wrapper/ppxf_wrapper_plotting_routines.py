from __future__ import print_function

import glob
from os import path
import matplotlib.pylab as plt

import numpy as np
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import astropy.units as u
from matplotlib.gridspec import GridSpec

def plot_kin_fit(ppw, text_loc=[0.98, 0.5], ax=None, xlabel=None, ylabel=None,
                           title=None, legend_loc='upper left', zoom_to_stars = False
                           ):
    if ax is None:
        fig, ax = plt.subplots(figsize=[13, 4])
        save_plot = True 
    else:
        save_plot = False # don't save if ax was given
    pp = ppw.pp
    fit_label = ppw.ssp_models
    spec_label = '{} spectrum'.format(ppw.instrument)
    ax.plot(pp.lam, pp.galaxy, c='k', label=spec_label, lw=1)
    ax.plot(pp.lam, pp.bestfit, c='orange', label='{0} fit'.format(fit_label), zorder=300, lw=1)

    res = pp.galaxy - pp.bestfit
    mn = np.min(pp.bestfit[pp.goodpixels])
    mn -= np.percentile(np.abs(res[pp.goodpixels]), 99)
    mx = np.max(pp.bestfit[pp.goodpixels])
    
    if (ppw.gas_fit) and (zoom_to_stars):
        print('ho')
        mn = np.min(pp.bestfit[pp.goodpixels]- pp.gas_bestfit[pp.goodpixels])
        mn -= np.percentile(np.abs(res[pp.goodpixels]), 99)
        mx = np.max(pp.bestfit[pp.goodpixels]- pp.gas_bestfit[pp.goodpixels])
    
    res += mn   # Offset residuals to avoid overlap
    mn1 = np.min(res[pp.goodpixels])
    
    if ppw.gas_fit:    
        ax.plot(pp.lam, pp.gas_bestfit + mn, c='darkred', zorder=2, lw=1)
        
    if ylabel is None:
        ax.set_ylabel("Relative Flux")
    else:
        ax.set_ylabel(ylabel)
    if xlabel is None:
        if ppw.templates.wavelength_unit == u.AA:
            ax.set_xlabel(r'$\rm{\lambda}$ ($\AA$)')
        else:
            ax.set_xlabel(r'$\rm{\lambda}$ ($\mu$m)')
    else:
        ax.set_xlabel(xlabel)
    ax.set_ylim([mn1, mx] + np.array([-0.05, 0.1])*(mx - mn1))

    w = np.flatnonzero(np.diff(pp.goodpixels) > 1)
    if w.size > 0:
        for wj in w:
            j = slice(pp.goodpixels[wj], pp.goodpixels[wj+1] + 1)
            xi = pp.lam[j]
            ax.axvspan(xi[0], xi[-1], color='grey', alpha=0.5)
        w = np.hstack([0, w, w + 1, -1])  # Add first and last point
    else:
        w = [0, -1]
    ax.plot(pp.lam, res, c='slateblue')
    if not title is None:
        ax.set_title(title)
        
    if not ppw.gas_fit:
        string = 'v = {0}'.format(np.round(
                    pp.sol[0], 1)) + r' km s$^{-1}$' + '\n' +r'$\sigma$ = ' + '{0}'.format(np.round(pp.sol[1], 1)) + r' km s$^{-1}$'
    else:
        string = r'$v_{\rm{stars}}$' + ' = {0}'.format(np.round(
            pp.sol[0][0], 1)) + r' km s$^{-1}$' + '\n' + \
            r'$v_{\rm{gas}}$' + ' = {0}'.format(np.round(
                pp.sol[1][0], 1)) + r' km s$^{-1}$'
    t = ax.text(text_loc[0], text_loc[1], string, color='black',     horizontalalignment='right',
                verticalalignment='center', fontsize=12,
                transform=ax.transAxes, backgroundcolor='w', zorder=350)
    t.set_bbox(dict(facecolor='w', alpha=0.8, edgecolor=None, linewidth=0))
    ax.legend(loc=legend_loc).set_zorder(1002)
    ax.set_xlim(pp.lam[0], pp.lam[-1])
    if (ppw.save_plots) and save_plot:
        plt.savefig(ppw.plot_out + ppw.plot_kin_title)

def plot_age_metal_grid(ppw, ax=None,  colorbar_position='top',
                        outdir = None, grid_plot_title='age_metal_grid'):
    if ax is None:
        fig, ax = plt.subplots(figsize=[6.5, 4])
        save_plot = ppw.save_plots
    else:
        save_plot = False
    pp = ppw.pp
    templates = ppw.templates
    
    weights = pp.weights[~pp.gas_component].reshape(
    templates.n_ages, templates.n_metal)/pp.weights[~pp.gas_component].sum()
    mean_age, mean_metal = templates.mean_age_metal(weights, quiet=True)
    
    xgrid = templates.age_grid
    ygrid = templates.metal_grid
    mean_age = np.sum(weights*xgrid)/np.sum(weights)
    mean_metal = np.sum(weights*ygrid)/np.sum(weights)

    std_age = np.sqrt(np.sum(weights)/(np.sum(weights)**2 - np.sum(weights**2))
                        * np.sum(weights * (xgrid - mean_age)**2))
    std_metal = np.sqrt(np.sum(weights)/(np.sum(weights)**2 - np.sum(weights**2))
                            * np.sum(weights * (ygrid - mean_metal)**2))
        
    x = xgrid[:, 0]
    y = ygrid[0, :]
    xb = (x[1:] + x[:-1])/2  # grid borders
    yb = (y[1:] + y[:-1])/2
    xb = np.hstack([1.5*x[0] - x[1]/2, xb, 1.5*x[-1] - x[-2]/2])
    yb = np.hstack([1.5*y[0] - y[1]/2, yb, 1.5*y[-1] - y[-2]/2])

    # pcolormesh() is used below to allow for irregular spacing in the
    # sampling of the stellar population parameters (e.g. metallicity)
    xlabel = "Age (Gyr)"
    ylabel = "[M/H]"
    pc = ax.pcolormesh(xb, yb, weights.T, cmap='bone_r', edgecolors='lightgray', lw=0.01)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(min(yb), max(yb))
    ax.set_xlim(min(xb), max(xb))
    t = ax.text(0.03, 0.08, 'Mean age: {0} Gyr'.format(np.round(mean_age, 1)) + '\n' + r'Mean [M/H]: {0}'.format(np.round(mean_metal, 2)), color='black',     horizontalalignment='left',
                verticalalignment='center', fontsize=12,
                transform=ax.transAxes, backgroundcolor='w')

    t.set_bbox(dict(facecolor='w', alpha=0.8, edgecolor=None, linewidth=0))
    ax.scatter(mean_age, mean_metal, color='k', marker='x')
    ax.errorbar(mean_age, mean_metal, xerr=std_age, yerr=std_metal, color='k', capsize=5)
    divider = make_axes_locatable(ax)
    if colorbar_position == 'top':
        cax = divider.append_axes("top", "1.5%", pad="1.5%")
        cbar = plt.colorbar(pc, cax=cax, orientation='horizontal')
        cax.xaxis.set_ticks_position('top')
        cax.xaxis.set_label_position('top')
    else:
        cax = divider.append_axes("right", "1.5%", pad="1.5%")
        cbar = plt.colorbar(pc, cax=cax)
    cbar.set_label(r'{0} weights'.format(ppw.ssp_models), fontsize=14)
    if save_plot:
        if outdir is None:
            outdir = ppw.plot_out
            print('Saving to {0}'.format(outdir))
        plt.saveplot(outdir + grid_plot_title)

def plot_pop_fit(ppw, ax0=None, ax1=None, xlabel=None, ylabel=None,
                           title=None, legend_loc='upper left', zoom_to_stars = False, outdir=None, save_plots=False):
    if ax0 is None:
        fig = plt.figure(figsize=[13, 3.5], constrained_layout=True)
        gs = GridSpec(1, 5, figure=fig)
        ax0 = fig.add_subplot(gs[:4])
        ax1 = fig.add_subplot(gs[4::])
    plot_kin_fit(ppw, ax=ax0, xlabel=xlabel, ylabel=ylabel, title=title, legend_loc=legend_loc, zoom_to_stars=zoom_to_stars)
    plot_age_metal_grid(ppw, ax=ax1, colorbar_position='top')
    
    for ax in [ax0, ax1]:
        ax.tick_params(labelsize=12)
        ax.xaxis.label.set_size(12)
        ax.yaxis.label.set_size(12)
    
    if (ppw.save_plots) or (save_plots):
        if outdir is None:
            outdir = ppw.plot_out
            plt.savefig(outdir + ppw.plot_pop_title, bbox_inches='tight')
        plt.savefig(outdir + ppw.plot_pop_title)


def get_bins(array, delta=4, nbins=30):
    bins = np.linspace(np.nanmean(array) - delta*np.nanstd(array),
                       np.nanmean(array) + delta*np.nanstd(array), nbins)
    return bins


def plot_hists(result, out_dir='./', kin_only=True):
    v = result[:, 0]
    sig = result[:, 1]
    if not kin_only:
        age = result[:, 2]
        metal = result[:, 3]

    n = len(v)
    vbins = get_bins(v)
    sbins = get_bins(sig)
    if kin_only:
        fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=[8, 4])
        ax0.hist(v, bins=vbins, facecolor='mediumblue', alpha=0.6, edgecolor='mediumblue')
        ax0.set_xlabel('v [km/s]')
        ax0.set_ylabel('# fits')
        ax1.hist(sig, bins=sbins, facecolor='darkred', alpha=0.6, edgecolor='darkred')
        ax1.set_xlabel(r'$\sigma$ [km/s]')
        ax1.set_ylabel('# fits')
        plt.tight_layout()
        plt.savefig(out_dir + 'Kin_hist_{0}_runs.png'.format(n), dpi=300)
    else:
        fig, (ax0, ax1, ax2, ax3) = plt.subplots(ncols=4, figsize=[16, 4])
        ax0.hist(v, bins=vbins, facecolor='mediumblue', alpha=0.6, edgecolor='mediumblue')
        ax0.set_xlabel('v [km/s]')
        ax0.set_ylabel('# fits')
        ax1.hist(sig, bins=sbins, facecolor='darkred', alpha=0.6, edgecolor='darkred')
        ax1.set_xlabel(r'$\sigma$ [km/s]')
        ax1.set_ylabel('# fits')
        ax2.hist(age, bins=get_bins(age), facecolor='orange', alpha=0.6, edgecolor='orange')
        ax2.set_xlabel(r'Age [Gyr]')
        ax2.set_ylabel('# fits')
        ax3.hist(metal, bins=get_bins(metal), facecolor='purple', alpha=0.6, edgecolor='purple')
        ax3.set_xlabel(r'[M/H]')
        ax3.set_ylabel('# fits')
        plt.tight_layout()
        plt.savefig(out_dir + 'Hists_{0}_runs.png'.format(n), dpi=300)


def plot_SFH(pp, templates, out_dir='./', save=False, plot=False, title='SFH.png', alpha_fit=False):
    if not alpha_fit:
        # assumes Etemplates library
        weights = pp.weights[:636].reshape(templates.n_ages, templates.n_metal)/pp.weights[:636].sum()
    else:
        weights = pp.weights.reshape(templates.n_ages, templates.n_metal, templates.n_alphas)/pp.weights.sum()

    ages = templates.age_grid[:, 0]
    summed_weights = np.zeros_like(ages)
    for i in range(len(ages)):  # loop over ages
        summi = np.sum(weights[i, :])
        summed_weights[i] = summi

    if plot:
        fig, ax = plt.subplots()
        ax.step(ages, summed_weights, color='darkred')
        ax.set_xlabel('Age [Gyr]')
        ax.set_ylabel('Mass fraction')
    if save:
        plt.savefig(out_dir + title, dpi=300)


# For alpha variable fits


def remove_abun_weight(pp, templates):
    weights = pp.weights[~pp.gas_component].reshape(
        templates.n_ages, templates.n_metal, templates.n_abuns)/pp.weights[~pp.gas_component].sum()

    s = np.shape(weights)
    weights_new = np.zeros([s[0], s[1]])

    for k in range(s[0]):
        for l in range(s[1]):
            other_weight = 0
            for m in range(s[2]):
                other_weight += weights[k, l, m]
            weights_new[k, l] = other_weight
    return weights_new


def remove_metal_weight(pp, templates):
    weights = pp.weights[~pp.gas_component].reshape(
        templates.n_ages, templates.n_metal, templates.n_abuns)/pp.weights[~pp.gas_component].sum()

    s = np.shape(weights)
    weights_new = np.zeros([s[0], s[2]])

    for k in range(s[0]):
        for l in range(s[2]):
            other_weight = 0
            for m in range(s[1]):
                other_weight += weights[k, m, l]
            weights_new[k, l] = other_weight
    return weights_new


def remove_age_weight(pp, templates):
    weights = pp.weights[~pp.gas_component].reshape(
        templates.n_ages, templates.n_metal, templates.n_abuns)/pp.weights[~pp.gas_component].sum()

    s = np.shape(weights)
    weights_new = np.zeros([s[1], s[2]])

    for k in range(s[1]):
        for l in range(s[2]):
            other_weight = 0
            for m in range(s[0]):
                other_weight += weights[m, k, l]
            weights_new[k, l] = other_weight
    return weights_new


def plot_pp_weights_abun(pp, templates, direct='./', title='Weights_alpha_fit.png', save=True, abun_prefix='alpha'):

    weights = pp.weights[~pp.gas_component].reshape(
        templates.n_ages, templates.n_metal, templates.n_abuns)/pp.weights[~pp.gas_component].sum()

    xgrid = templates.age_grid
    ygrid = templates.metal_grid
    zgrid = templates.abun_grid
    mean_age = np.sum(weights*xgrid)/np.sum(weights)
    mean_metal = np.sum(weights*ygrid)/np.sum(weights)
    mean_abun = np.sum(weights*zgrid)/np.sum(weights)

    std_age = np.sqrt(np.sum(weights)/(np.sum(weights)**2 - np.sum(weights**2))
                      * np.sum(weights * (xgrid - mean_age)**2))
    std_metal = np.sqrt(np.sum(weights)/(np.sum(weights)**2 - np.sum(weights**2))
                        * np.sum(weights * (ygrid - mean_metal)**2))
    std_abun = np.sqrt(np.sum(weights)/(np.sum(weights)**2 - np.sum(weights**2))
                       * np.sum(weights * (zgrid - mean_abun)**2))

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(7, 8))

    # plot age vs metal
    x = xgrid[:, 0, 0]
    y = ygrid[0, :, 0]
    z = zgrid[0, 0, :]
    xb = (x[1:] + x[:-1])/2  # grid borders
    yb = (y[1:] + y[:-1])/2
    zb = (z[1:] + z[:-1])/2
    xb = np.hstack([1.5*x[0] - x[1]/2, xb, 1.5*x[-1] - x[-2]/2])
    yb = np.hstack([1.5*y[0] - y[1]/2, yb, 1.5*y[-1] - y[-2]/2])
    zb = np.hstack([1.5*z[0] - z[1]/2, zb, 1.5*z[-1] - z[-2]/2])
    # pcolormesh() is used below to allow for irregular spacing in the
    # sampling of the stellar population parameters (e.g. metallicity)
    ax = ax0
    xlabel = "Age (Gyr)"
    ylabel = "[M/H]"
    age_metal_weights = remove_abun_weight(pp, templates)
    pc = ax.pcolormesh(xb, yb, age_metal_weights.T, cmap='bone_r', edgecolors='lightgray', lw=0.01)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(min(yb), max(yb))
    ax.set_xlim(min(xb), max(xb))
    ax.text(0.03, 0.09, 'Mean age: {0} Gyr, Mean [M/H]: {1}'.format(np.round(mean_age, 1), np.round(mean_metal, 2)), color='black',     horizontalalignment='left',
            verticalalignment='center', fontsize=12,
            transform=ax.transAxes, backgroundcolor='w')
    ax.scatter(mean_age, mean_metal, color='k', marker='x')
    ax.errorbar(mean_age, mean_metal, xerr=std_age, yerr=std_metal, color='k', capsize=5)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "1.5%", pad="1.5%")
    cbar = plt.colorbar(pc, cax=cax)
    cbar.set_label(r'weights')

    # Age vs alpha
    x = xgrid[:, 0, 0]
    y = zgrid[0, 0, :]
    xb = (x[1:] + x[:-1])/2  # grid borders
    yb = (y[1:] + y[:-1])/2
    xb = np.hstack([1.5*x[0] - x[1]/2, xb, 1.5*x[-1] - x[-2]/2])
    yb = np.hstack([1.5*y[0] - y[1]/2, yb, 1.5*y[-1] - y[-2]/2])

    # pcolormesh() is used below to allow for irregular spacing in the
    # sampling of the stellar population parameters (e.g. metallicity)
    ax = ax1
    xlabel = "Age (Gyr)"
    if abun_prefix == 'alpha':
        ylabel = r"[$\alpha$/Fe]"
    else:
        ylabel = r"[{0}/Fe]".format(abun_prefix)
    age_abun_weights = remove_metal_weight(pp, templates)
    pc = ax.pcolormesh(xb, yb, age_abun_weights.T,
                       cmap='bone_r', edgecolors='lightgray', lw=0.01)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(min(yb), max(yb))
    ax.set_xlim(min(xb), max(xb))
    ax.text(0.03, 0.09, r'Mean age: {0} Gyr, {1}: {2}'.format(np.round(mean_age, 1), ylabel, np.round(mean_abun, 2)), color='black',     horizontalalignment='left',
            verticalalignment='center', fontsize=12,
            transform=ax.transAxes, backgroundcolor='w')
    ax.scatter(mean_age, mean_abun, color='k', marker='x')
    ax.errorbar(mean_age, mean_abun, xerr=std_age, yerr=std_abun, color='k', capsize=5)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "1.5%", pad="1.5%")
    cbar = plt.colorbar(pc, cax=cax)
    cbar.set_label(r'weights')

    # metal vs alpha
    x = ygrid[0, :, 0]
    y = zgrid[0, 0, :]
    xb = (x[1:] + x[:-1])/2  # grid borders
    yb = (y[1:] + y[:-1])/2
    xb = np.hstack([1.5*x[0] - x[1]/2, xb, 1.5*x[-1] - x[-2]/2])
    yb = np.hstack([1.5*y[0] - y[1]/2, yb, 1.5*y[-1] - y[-2]/2])

    # pcolormesh() is used below to allow for irregular spacing in the
    # sampling of the stellar population parameters (e.g. metallicity)
    ax = ax2
    xlabel = "[M/H]"
    #ylabel = r"[$\alpha$/Fe]"
    metal_alpha_weights = remove_age_weight(pp, templates)
    pc = ax.pcolormesh(xb, yb, metal_alpha_weights.T, cmap='bone_r',
                       edgecolors='lightgray', lw=0.01)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(min(yb), max(yb))
    ax.set_xlim(min(xb), max(xb))
    ax.text(0.03, 0.09, r'Mean [M/H]: {0} [dex], Mean {1}: {2}'.format(np.round(mean_metal, 2), ylabel, np.round(mean_abun, 2)), color='black',     horizontalalignment='left',
            verticalalignment='center', fontsize=12,
            transform=ax.transAxes, backgroundcolor='w')
    ax.scatter(mean_metal, mean_abun, color='k', marker='x')
    ax.errorbar(mean_metal, mean_abun, xerr=std_age, yerr=std_abun, color='k', capsize=5)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "1.5%", pad="1.5%")
    cbar = plt.colorbar(pc, cax=cax)
    cbar.set_label(r'weights')

    plt.tight_layout()

    if save:
        plt.savefig(direct + title, dpi=300)


def plot_pp_pops_abun(pp, templates, direct='./', title='Stellar_pop_abun_fit.png', save=True, abun_prefix='alpha'):
    weights = pp.weights[~pp.gas_component].reshape(
        templates.n_ages, templates.n_metal, templates.n_abuns)/pp.weights[~pp.gas_component].sum()

    xgrid = templates.age_grid
    ygrid = templates.metal_grid
    zgrid = templates.abun_grid
    mean_age = np.sum(weights*xgrid)/np.sum(weights)
    mean_metal = np.sum(weights*ygrid)/np.sum(weights)
    mean_abun = np.sum(weights*zgrid)/np.sum(weights)

    std_age = np.sqrt(np.sum(weights)/(np.sum(weights)**2 - np.sum(weights**2))
                      * np.sum(weights * (xgrid - mean_age)**2))
    std_metal = np.sqrt(np.sum(weights)/(np.sum(weights)**2 - np.sum(weights**2))
                        * np.sum(weights * (ygrid - mean_metal)**2))
    std_abun = np.sqrt(np.sum(weights)/(np.sum(weights)**2 - np.sum(weights**2))
                       * np.sum(weights * (zgrid - mean_abun)**2))

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 6))
    # Plot spectrum + fit + residual
    x = pp.lam

    res = pp.galaxy - pp.bestfit
    mn = np.min(pp.bestfit[pp.goodpixels])
    mn -= np.percentile(np.abs(res[pp.goodpixels]), 99)
    mx = np.max(pp.bestfit[pp.goodpixels])
    res += mn   # Offset residuals to avoid overlap
    mn1 = np.min(res[pp.goodpixels])
    ax0.set_ylabel("Relative Flux")
    ax0.set_xlabel(r'$\rm{\lambda}$ [$\AA$]')
    ax0.set_xlim(x[0], x[-1])
    ax0.set_ylim([mn1, mx] + np.array([-0.05, 0.1])*(mx - mn1))
    w = np.flatnonzero(np.diff(pp.goodpixels) > 1)
    if w.size > 0:
        for wj in w:
            j = slice(pp.goodpixels[wj], pp.goodpixels[wj+1] + 1)
            ax0.plot(x[j], res[j], 'mediumblue', lw=1)
            xi = x[j]
            resi = res[j]
            ax0.axvspan(xi[0], xi[-1], color='grey', alpha=0.5)
        w = np.hstack([0, w, w + 1, -1])  # Add first and last point
    else:
        w = [0, -1]
    ax0.step(x, pp.galaxy, 'k', linewidth=1)
    ax0.plot(x[pp.goodpixels], res[pp.goodpixels], 'd',
             color='LimeGreen', mec='c', ms=1)
    ax0.plot(x, np.zeros_like(x)+mn, lw=2, ls=':')
    ax0.plot(x, pp.bestfit, 'orangered', linewidth=1)

    x = xgrid[:, 0, 0]
    y = ygrid[0, :, 0]
    xb = (x[1:] + x[:-1])/2  # grid borders
    yb = (y[1:] + y[:-1])/2
    xb = np.hstack([1.5*x[0] - x[1]/2, xb, 1.5*x[-1] - x[-2]/2])
    yb = np.hstack([1.5*y[0] - y[1]/2, yb, 1.5*y[-1] - y[-2]/2])

    # pcolormesh() is used below to allow for irregular spacing in the
    # sampling of the stellar population parameters (e.g. metallicity)
    weights_new = remove_abun_weight(pp, templates)
    xlabel = "Age (Gyr)"
    ylabel = "[M/H]"
    pc = ax1.pcolormesh(xb, yb, weights_new.T, cmap='bone_r', edgecolors='lightgray', lw=0.01)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_ylim(min(yb), max(yb))
    ax1.set_xlim(min(xb), max(xb))
    ax1.text(0.03, 0.09, 'Mean age: {0} Gyr, Mean [M/H]: {1}'.format(np.round(mean_age, 1), np.round(mean_metal, 2)), color='black',     horizontalalignment='left',
             verticalalignment='center', fontsize=12,
             transform=ax1.transAxes, backgroundcolor='w')
    ax1.scatter(mean_age, mean_metal, color='k', marker='x')
    ax1.errorbar(mean_age, mean_metal, xerr=std_age, yerr=std_metal, color='k', capsize=5)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", "1.5%", pad="1.5%")
    cbar = plt.colorbar(pc, cax=cax)
    cbar.set_label(r'weights')
    plt.tight_layout()
    if save:
        plt.savefig(direct + title, dpi=300)


def plot_pp_pops_abun_full_weights(pp, templates, direct='./', title='Stellar_pop_abun_fit_with_weights.png', save=True, abun_prefix='alpha'):
    weights = pp.weights[~pp.gas_component].reshape(
        templates.n_ages, templates.n_metal, templates.n_abuns)/pp.weights[~pp.gas_component].sum()

    xgrid = templates.age_grid
    ygrid = templates.metal_grid
    zgrid = templates.abun_grid
    mean_age = np.sum(weights*xgrid)/np.sum(weights)
    mean_metal = np.sum(weights*ygrid)/np.sum(weights)
    mean_abun = np.sum(weights*zgrid)/np.sum(weights)

    std_age = np.sqrt(np.sum(weights)/(np.sum(weights)**2 - np.sum(weights**2))
                      * np.sum(weights * (xgrid - mean_age)**2))
    std_metal = np.sqrt(np.sum(weights)/(np.sum(weights)**2 - np.sum(weights**2))
                        * np.sum(weights * (ygrid - mean_metal)**2))
    std_abun = np.sqrt(np.sum(weights)/(np.sum(weights)**2 - np.sum(weights**2))
                       * np.sum(weights * (zgrid - mean_abun)**2))

    fig = plt.figure(figsize=(10, 6))
    ax0 = plt.subplot2grid((2, 3), (0, 0), colspan=3)
    ax1 = plt.subplot2grid((2, 3), (1, 0))
    ax2 = plt.subplot2grid((2, 3), (1, 1))
    ax3 = plt.subplot2grid((2, 3), (1, 2))
    # Plot spectrum + fit + residual
    x = pp.lam

    res = pp.galaxy - pp.bestfit
    mn = np.min(pp.bestfit[pp.goodpixels])
    mn -= np.percentile(np.abs(res[pp.goodpixels]), 99)
    mx = np.max(pp.bestfit[pp.goodpixels])
    res += mn   # Offset residuals to avoid overlap
    mn1 = np.min(res[pp.goodpixels])
    ax0.set_ylabel("Relative Flux")
    ax0.set_xlabel(r'$\rm{\lambda}$ [$\AA$]')
    ax0.set_xlim(x[0], x[-1])
    ax0.set_ylim([mn1, mx] + np.array([-0.05, 0.1])*(mx - mn1))
    w = np.flatnonzero(np.diff(pp.goodpixels) > 1)
    if w.size > 0:
        for wj in w:
            j = slice(pp.goodpixels[wj], pp.goodpixels[wj+1] + 1)
            ax0.plot(x[j], res[j], 'mediumblue', lw=1)
            xi = x[j]
            resi = res[j]
            ax0.axvspan(xi[0], xi[-1], color='grey', alpha=0.5)
        w = np.hstack([0, w, w + 1, -1])  # Add first and last point
    else:
        w = [0, -1]
    ax0.step(x, pp.galaxy, 'k', linewidth=1)
    ax0.plot(x[pp.goodpixels], res[pp.goodpixels], 'd',
             color='LimeGreen', mec='c', ms=1)
    ax0.plot(x, np.zeros_like(x)+mn, lw=2, ls=':')
    ax0.plot(x, pp.bestfit, 'orangered', linewidth=1)

    x = xgrid[:, 0, 0]
    y = ygrid[0, :, 0]
    z = zgrid[0, 0, :]
    xb = (x[1:] + x[:-1])/2  # grid borders
    yb = (y[1:] + y[:-1])/2
    zb = (z[1:] + z[:-1])/2
    xb = np.hstack([1.5*x[0] - x[1]/2, xb, 1.5*x[-1] - x[-2]/2])
    yb = np.hstack([1.5*y[0] - y[1]/2, yb, 1.5*y[-1] - y[-2]/2])
    zb = np.hstack([1.5*z[0] - z[1]/2, zb, 1.5*z[-1] - z[-2]/2])

    # ax1: age vs metallicities
    ax = ax1
    xlabel = "Age (Gyr)"
    ylabel = "[M/H]"
    age_metal_weights = remove_abun_weight(pp, templates)
    pc = ax.pcolormesh(xb, yb, age_metal_weights.T, cmap='bone_r', edgecolors='lightgray', lw=0.01)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(min(yb), max(yb))
    ax.set_xlim(min(xb), max(xb))
    ax.text(0.03, 0.13, 'Mean age: {0} Gyr\nMean [M/H]: {1} dex'.format(np.round(mean_age, 1), np.round(mean_metal, 2)), color='black',     horizontalalignment='left',
            verticalalignment='center', fontsize=9,
            transform=ax.transAxes, backgroundcolor='w')
    ax.scatter(mean_age, mean_metal, color='k', marker='x')
    ax.errorbar(mean_age, mean_metal, xerr=std_age, yerr=std_metal, color='k', capsize=5)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", "2.5%", pad="1.5%")
    cbar = plt.colorbar(pc, cax=cax, orientation='horizontal', ticklocation='top')
    cbar.set_label(r'Weights')
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.update_ticks()

    # Age vs alpha
    x = xgrid[:, 0, 0]
    y = zgrid[0, 0, :]
    xb = (x[1:] + x[:-1])/2  # grid borders
    yb = (y[1:] + y[:-1])/2
    xb = np.hstack([1.5*x[0] - x[1]/2, xb, 1.5*x[-1] - x[-2]/2])
    yb = np.hstack([1.5*y[0] - y[1]/2, yb, 1.5*y[-1] - y[-2]/2])

    # pcolormesh() is used below to allow for irregular spacing in the
    # sampling of the stellar population parameters (e.g. metallicity)
    ax = ax2
    xlabel = "Age (Gyr)"
    if abun_prefix == 'alpha':
        ylabel = r"[$\alpha$/Fe]"
    else:
        ylabel = r'[{0}/Fe]'.format(abun_prefix)
    age_abun_weights = remove_metal_weight(pp, templates)
    pc = ax.pcolormesh(xb, yb, age_abun_weights.T,
                       cmap='bone_r', edgecolors='lightgray', lw=0.01)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(min(yb), max(yb))
    ax.set_xlim(min(xb), max(xb))
    ax.text(0.03, 0.13, 'Mean age: {0} Gyr'.format(np.round(mean_age, 1)) + '\n' + r' Mean {0}: {1} dex'.format(ylabel, np.round(mean_abun, 2)), color='black',     horizontalalignment='left',
            verticalalignment='center', fontsize=9,
            transform=ax.transAxes, backgroundcolor='w')
    ax.scatter(mean_age, mean_abun, color='k', marker='x')
    ax.errorbar(mean_age, mean_abun, xerr=std_age, yerr=std_abun, color='k', capsize=5)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", "2.5%", pad="1.5%")
    cbar = plt.colorbar(pc, cax=cax, orientation='horizontal', ticklocation='top')
    cbar.set_label(r'Weights')
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.update_ticks()

    # metal vs alpha
    x = ygrid[0, :, 0]
    y = zgrid[0, 0, :]
    xb = (x[1:] + x[:-1])/2  # grid borders
    yb = (y[1:] + y[:-1])/2
    xb = np.hstack([1.5*x[0] - x[1]/2, xb, 1.5*x[-1] - x[-2]/2])
    yb = np.hstack([1.5*y[0] - y[1]/2, yb, 1.5*y[-1] - y[-2]/2])

    # pcolormesh() is used below to allow for irregular spacing in the
    # sampling of the stellar population parameters (e.g. metallicity)
    ax = ax3
    xlabel = "[M/H]"
    metal_abun_weights = remove_age_weight(pp, templates)
    pc = ax.pcolormesh(xb, yb, metal_abun_weights.T, cmap='bone_r',
                       edgecolors='lightgray', lw=0.01)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(min(yb), max(yb))
    ax.set_xlim(min(xb), max(xb))
    ax.text(0.03, 0.13, r'Mean [M/H]: {0} dex'.format(np.round(mean_metal, 2)) + '\n' + r'Mean {0}: {1} dex'.format(ylabel, np.round(mean_abun, 2)), color='black',     horizontalalignment='left',
            verticalalignment='center', fontsize=9,
            transform=ax.transAxes, backgroundcolor='w')
    ax.scatter(mean_metal, mean_abun, color='k', marker='x')
    ax.errorbar(mean_metal, mean_abun, xerr=std_age, yerr=std_abun, color='k', capsize=5)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", "2.5%", pad="1.5%")
    cbar = plt.colorbar(pc, cax=cax, orientation='horizontal', ticklocation='top')
    cbar.set_label(r'Weights')
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.update_ticks()

    plt.tight_layout()
    if save:
        plt.savefig(direct + title, dpi=300)
