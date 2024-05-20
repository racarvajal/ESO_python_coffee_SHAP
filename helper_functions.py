#!/usr/bin/env python

# File with some
# functions

# Initial imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mtick

def plot_polynomial_degrees(polynomial, coeffs, degree=5, xlims=[-2, 2], npoints=20):
    fig             = plt.figure(figsize=(12, 4), constrained_layout=False)

    grid            = fig.add_gridspec(ncols=degree+1, nrows=1, width_ratios=[1]*(degree + 1),
                                    height_ratios=[1], hspace=0.0, wspace=0.0)
    axs             = {}

    axs[0] = fig.add_subplot(grid[0, 0])
    for count in np.arange(degree + 1)[1:]:
        axs[count] = fig.add_subplot(grid[0, count], sharey=axs[0])

    zeros = np.zeros(npoints)
    value = np.linspace(xlims[0], xlims[1], npoints)

    funcs = []
    for count in np.arange(degree + 1):
        funcs_tmp        = np.array([zeros] * (degree + 1))
        funcs_tmp[count] = value
        funcs_tmp        = funcs_tmp.reshape((degree + 1, npoints))
        funcs.append(funcs_tmp)

    for count in np.arange(degree + 1):
        axs[count].plot(value, polynomial(funcs[count].T), ls='-', marker='o', ms=8, color='k')
        axs[count].set_title(rf'$\mathrm{{Degree}} ~ {count}$', fontsize=22)
        axs[count].set_xlabel(rf'$\mathrm{{x}}_{count}$', fontsize=22)

    axs[0].set_ylabel(r'$\mathrm{f (x)}$', fontsize=22)

    title = rf'$\mathrm{{f (x) = {coeffs[0]}}}$'
    for count in np.arange(degree + 1)[1:]:
        title += rf'${coeffs[count]:+} x^{count}_{count}$'

    # fig.suptitle(r'$\mathrm{f (x) = x + 2x^{2} - 3x^{3} + 0.8x^{4} - 0.5x^{5}}$', fontsize=22)
    fig.suptitle(title, fontsize=22)

    for count in np.arange(degree + 1):
        axs[count].tick_params(which='both', top=True, right=True, direction='in')
        axs[count].tick_params(axis='both', which='minor', labelsize=24.5)
        axs[count].tick_params(axis='both', which='major', labelsize=24.5)
        axs[count].tick_params(which='major', length=8, width=1.5)
        axs[count].tick_params(which='minor', length=4, width=1.5)
        plt.setp(axs[count].spines.values(), linewidth=2.5)
        plt.setp(axs[count].spines.values(), linewidth=2.5)
    for count in np.arange(degree + 1)[1:]:
        plt.setp(axs[count].get_yticklabels(), visible=False)
    plt.tight_layout()

    plt.show()


def plot_redshifts(true_z, pred_z, cmap='cividis_r'):
    fig             = plt.figure(figsize=(7.5, 6.0))
    ax1             = fig.add_subplot(111, xscale='log', yscale='log')

    min_for_range = np.nanmin([np.nanmin(1 + true_z), np.nanmin(1 + pred_z)])
    max_for_range = np.nanmax([np.nanmax(1 + true_z), np.nanmax(1 + pred_z)])
    bins_z        = np.logspace(np.log10(min_for_range), np.log10(max_for_range), num=40)       

    norm = mcolors.LogNorm()

    _, _, _, hist_sources = ax1.hist2d((1 + true_z), (1 + pred_z), 
                                        bins=bins_z, cmin=1, cmap=cmap, norm=norm)
        
    ax1.axline((2., 2.), (3., 3.), ls='--', marker=None, c='k', alpha=0.8, lw=3.0, zorder=20)

    clb = plt.colorbar(hist_sources, extend='neither', norm=norm, ticks=mtick.MaxNLocator(integer=True))
    clb.ax.yaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=False))
    clb.ax.tick_params(labelsize=20)
    clb.outline.set_linewidth(2.5)
    clb.ax.set_ylabel(r'$\mathrm{Sources ~ per ~ bin}$', size=28)

    ax1.set_xlabel('$1 + \mathit{z}_{\mathrm{True}}$', fontsize=32)
    ax1.set_ylabel('$1 + \mathit{z}_{\mathrm{Predicted}}$', fontsize=32)
    ax1.tick_params(which='both', top=True, right=True, direction='in')
    ax1.tick_params(axis='both', which='minor', labelsize=24.5)
    ax1.tick_params(axis='both', which='major', labelsize=24.5)
    ax1.tick_params(which='major', length=8, width=1.5)
    ax1.tick_params(which='minor', length=4, width=1.5)
    ax1.xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
    ax1.yaxis.set_major_locator(mtick.MaxNLocator(integer=True))
    ax1.xaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=False))
    ax1.yaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=False))
    plt.setp(ax1.spines.values(), linewidth=2.5)
    plt.setp(ax1.spines.values(), linewidth=2.5)
    ax1.set_xlim(left=1.01, right=np.ceil(max_for_range))
    ax1.set_ylim(bottom=1.01, top=np.ceil(max_for_range))
    plt.tight_layout()
    plt.show()

def tabular_shap_vals(explanation):
    col_names           = explanation.feature_names
    mean_abs_SHAP_coefs = np.mean(np.abs(explanation.values), axis=0)
    mean_abs_SHAP_df    = pd.DataFrame({'Feature': col_names, 'Mean abs SHAP': mean_abs_SHAP_coefs})
    mean_abs_SHAP_df    = mean_abs_SHAP_df.sort_values(by='Mean abs SHAP',
                                                        ascending=False).reset_index(drop=True)
    mean_abs_SHAP_df.loc[:, 'Mean abs SHAP %'] = (mean_abs_SHAP_df.loc[:, 'Mean abs SHAP'] /
                                                    mean_abs_SHAP_df.loc[:, 'Mean abs SHAP'].sum()) * 100
    mean_abs_SHAP_df['Cumulative sum %'] = mean_abs_SHAP_df.loc[:, 'Mean abs SHAP %'].cumsum()
    return mean_abs_SHAP_df
