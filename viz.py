# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.patches as mpatches
import vip_hci as vip
import itertools


def hist_normal(data, bins=10, alpha=1., ax=None):
    from scipy.stats import norm
    
    if ax is None:
        ax = plt.axes()

    n, bins, patches = ax.hist(data, bins=bins, normed=1, alpha=alpha)

    (mu, sigma) = norm.fit(data)
    y = mlab.normpdf(bins, mu, sigma)
    l = ax.plot(bins, y, '--', linewidth=2)
    

def compare_distributions(data, bins, labels='', ax=None):
    from scipy.stats import norm
    import itertools
    
    if ax is None:
        ax = plt.axes()
        
    dmin = np.min([np.min(d) for d in data])
    dmax = np.max([np.max(d) for d in data])

    bins = np.linspace(dmin, dmax, bins)

    ax.hist(data, bins=bins, normed=1,
            label=labels, alpha=.7)
    
    ax.set_prop_cycle(None)
    
    for d in data:
        (mu, sigma) = norm.fit(d)
        y = mlab.normpdf(bins, mu, sigma)
        l = plt.plot(bins, y, '--', linewidth=3)

    
def auc_heatmap(aucs, xaxis=True, yaxis=True, colorbar=True, ax=None):
    if ax is None:
        ax = plt.axes()
    
    heatmap = ax.pcolor(aucs)
    heatmap.set_clim([.5, 1.])

    if xaxis:
        ax.xaxis.set_ticks(np.arange(0.5, len(aucs.columns), 1))
        ax.xaxis.set_ticklabels(aucs.columns)
#         ax.xaxis.set_label_text('Distance from\nstar [$\lambda/D$]')
    else:
        ax.xaxis.set_ticklabels([])
    
    if yaxis:
        ax.yaxis.set_ticks(np.arange(0.5, len(aucs.index), 1))
        ax.yaxis.set_ticklabels(aucs.index)
#         ax.yaxis.set_label_text('Injected planet flux')
    else:
        ax.yaxis.set_ticklabels([])
        
    ax.xaxis.set_tick_params(length=0.)
    ax.yaxis.set_tick_params(length=0.)
    
    if colorbar:
        plt.colorbar(heatmap, ax=ax)
    
    return heatmap


def auc_diff_heatmap(aucs1, aucs2, lim=None, xaxis=True, yaxis=True, colorbar=True, ax=None):
    if ax is None:
        ax = plt.axes()
    
    diff = aucs1 - aucs2
        
    if lim is None:
        lim = np.abs(diff).max().max()
    
    heatmap = ax.pcolor(diff, cmap='coolwarm')
    heatmap.set_clim([-lim, lim])

    if xaxis:
        ax.xaxis.set_ticks(np.arange(0.5, len(diff.columns), 1))
        ax.xaxis.set_ticklabels(diff.columns)
#         ax.xaxis.set_label_text('Distance from\nstar [$\lambda/D$]')
    else:
        ax.xaxis.set_ticklabels([])
    
    if yaxis:
        ax.yaxis.set_ticks(np.arange(0.5, len(diff.index), 1))
        ax.yaxis.set_ticklabels(diff.index)
#         ax.yaxis.set_label_text('Injected planet flux')
    else:
        ax.yaxis.set_ticklabels([])
        
    ax.xaxis.set_tick_params(length=0.)
    ax.yaxis.set_tick_params(length=0.)
    
    if colorbar:
        plt.colorbar(heatmap, ax=ax)
    
    return heatmap


def plot_frame_detections(frame, detections, fwhm, ax=None):
    if ax is None:
        ax = plt.gca()
    
    cy, cx = vip.var.frame_center(frame)
    
    im = ax.imshow(frame)
    for (y, x) in zip(detections.y + cy, detections.x + cx):
        circle = mpatches.Circle(
            (x, y), fwhm/2,
            facecolor='none', edgecolor='red', linewidth=2.)
        ax.add_artist(circle)
    
    return im

        
def plot_frame_detections_from_results(results, fwhm, score_column, ax=None):
    from perf_assess import results_detections
    plot_frame_detections(
        results[1][score_column],
        results_detections(results, fwhm)[1],
        fwhm,
        ax=ax
    )


def roc_curves(perfs, label_format, ax=None, legend=True):
    if ax is None:
        ax = plt.gca()
    
    if type(label_format) == list:
        labels = label_format
    else:
        labels = [label_format.format(**perf) for perf in perfs]
    
    for perf, label in zip(perfs, labels):
        ax.plot(perf.fp, perf.tpr, label=label)
    
    if legend:
        ax.legend()


def roc_curves_multiple(perfs_list, label_format, n_cols=5, legend=True):
    n = len(perfs_list[0])
    n_cols = min(n_cols, n)
    n_rows = int(np.ceil(n/float(n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3))
    
    if type(axes) != np.ndarray:
        axes = np.array([axes])
    
    if axes.ndim == 1:
        axes[0].set_ylabel("TPR")
        for ax in axes:
            ax.set_xlabel("Frame total FP")
    
    elif axes.ndim == 2:
        for ax in axes[:, 0]:
            ax.set_ylabel("TPR")
        for ax in axes[-1, :]:
            ax.set_xlabel("Frame total FP")
    
    for ax, perfs in zip(axes.flat, zip(*perfs_list)):
        ax.set_title('{sep_from:1.1f}--{sep_to:1.1f} $\\lambda/D$'.format(**perfs[0]))

        roc_curves(perfs, label_format, ax=ax, legend=legend)

        if legend:
            ax.legend()

    return fig, axes


def aac_plot(perfs_list, label_format, colors=None, ax=None, legend=True,
             markers=('o', 'v', '^', '<', '>', 's', 'P', 'X', 'd')):
    if ax is None:
        ax = plt.gca()
    
    if colors is None:
        colors = [None] * len(perfs_list)
    
    if type(label_format) == list:
        labels = label_format
    else:
        labels = [label_format.format(**perfs[0]) for perfs in perfs_list]
    
    marker = itertools.cycle(markers)

    for perfs, color, label in zip(perfs_list, colors, labels):
        xs = [(perf.sep_from + perf.sep_to)/2 for perf in perfs]
        ys = [perf.aac for perf in perfs]
        plt.plot(xs, ys,
                 marker=marker.next(), linestyle='--', color=color,
                 label=label)

    ax.set_xlabel('Separation [$\\lambda/D$]')
    ax.set_ylabel('Area above the FROC curve')
    ax.set_xticks(xs)
    ax.set_xticklabels(["{sep_from:1.1f}--{sep_to:1.1f}".format(**perf) for perf in perfs])

    if legend:
        ax.legend(loc=legend if type(legend) == str else 'best')
