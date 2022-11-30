#!/usr/bin/env python

import glob
import os
import sys

import astropy.io.fits as pf
import matplotlib.pyplot as plt
import numpy as np


cumulative = True
colors = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:grey",
    "tab:olive",
    "tab:cyan",
    "black",
    "gold",
]

if len(sys.argv) == 1:
    raise RuntimeError(f"Usage: {sys.argv[0]} <output dir1> [<output_dir2>]")

outdirs = sys.argv[1:]

nrow1, ncol1 = 4, 4
fig1 = plt.figure(figsize=[ncol1 * 4, nrow1 * 4])
axes1 = None

nrow2, ncol2 = 4, 8
fig2 = plt.figure(figsize=[ncol2 * 4, nrow2 * 4])
axes2 = None

for fig in fig1, fig2:
    if cumulative:
        fig.suptitle("Cumulative parameter values")
    else:
        fig.suptitle("Non-cumulative parameter values")

dets = None

all_params = None
params = None
harmonics = None

axes1 = []
axes2 = []

for outdir, linestyle in zip(outdirs, ["-", "--", ":"]):
    param_fits = {}
    harmonic_fits = {}

    fnames = glob.glob(f"{outdir}/baselines_iter??.fits")
    for fname in sorted(fnames):
        print(fname)
        hdulist = pf.open(fname, "readonly")
        if all_params is None:
            all_params = [hdu.header["extname"] for hdu in hdulist[1:]]
            params = []
            harmonics = []
            for param in all_params:
                if param.startswith("spin_harmonic"):
                    harmonics.append(param)
                else:
                    params.append(param)
        if dets is None:
            dets = [col.name for col in hdulist[params[0]].columns]
        for iparam, param in enumerate(params):
            if param not in param_fits:
                param_fits[param] = {}
                for det in dets:
                    param_fits[param][det] = []
            if param not in hdulist:
                continue
            for det in dets:
                param_fits[param][det].append(hdulist[param].data[det])
        for iparam, param in enumerate(harmonics):
            if param not in harmonic_fits:
                harmonic_fits[param] = {}
                for det in dets:
                    harmonic_fits[param][det] = []
            if param not in hdulist:
                continue
            for det in dets:
                harmonic_fits[param][det].append(hdulist[param].data[det])

    print(params)
    #nparam = len(params)
    #nharmonic = len(harmonics)

    for iparam, param in enumerate(params):
        if len(axes1) == iparam:
            axes1.append(fig1.add_subplot(nrow1, ncol1, iparam + 1))
        ax = axes1[iparam]
        ax.set_title(param)
        for det, color in zip(dets, colors):
            values = param_fits[param][det]
            limits = []
            total = 0
            if cumulative and param not in [
                    "offset", "pol0", "pol1", "pol2", "pol3", "pol4"
            ]:
                for i in range(1, len(values)):
                    values[i] += values[i - 1]
                    total += len(values[i - 1])
                    if total > 10:
                        limits.append(total)
            if len(values) == 0:
                continue
            values = np.hstack(values)
            if np.all(values == 0):
                continue
            ax.plot(values, label=f"{det} {outdir}", color=color, linestyle=linestyle)
        for limit in limits:
            ax.axvline(limit, color="k", linestyle="--")

    for iparam, param in enumerate(harmonics):
        if len(axes2) == iparam:
            axes2.append(fig2.add_subplot(nrow2, ncol2, iparam + 1))
        ax = axes2[iparam]
        ax.set_title(param)
        for det, color in zip(dets, colors):
            values = harmonic_fits[param][det]
            if cumulative:
                for i in range(1, len(values)):
                    values[i] += values[i - 1]
            values = np.hstack(values)
            if np.all(values == 0):
                continue
            ax.plot(values, label=f"{det} {outdir}", color=color, linestyle=linestyle)

axes1[-1].legend(bbox_to_anchor=(1,1), loc="upper left")
axes2[-1].legend(bbox_to_anchor=(1,1), loc="upper left")

fig1.savefig("parameter_fits.png")
fig2.savefig("harmonic_fits.png")

fig1.tight_layout()
fig2.tight_layout()
