#!/usr/bin/env python

from __future__ import print_function

import os

import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator
import requests


try:
    os.makedirs("results")
except os.error:
    pass


sdss_url = "http://api.sdss3.org"


def find_star(ra, dec, radius):
    # Build request.
    url = sdss_url + "/spectrumQuery"
    pars = {"ra": "{0:f}d".format(ra), "dec": "{0:f}d".format(dec),
            "radius": float(radius)}

    # Submit request.
    r = requests.get(url, params=pars)
    if r.status_code != requests.codes.ok:
        r.raise_for_status()

    # Get the contents of the response.
    data = r.json
    assert len(data) >= 1

    return data[0]


def get_star(ra, dec, radius=0.005):
    # Find the SDSS ID.
    sdss_id = find_star(ra, dec, radius)

    # Build request.
    url = sdss_url + "/spectrum"
    pars = {"id": sdss_id, "format": "json", }

    # Submit request.
    r = requests.get(url, params=pars)
    if r.status_code != requests.codes.ok:
        r.raise_for_status()

    # Get the contents of the response.
    data = r.json
    assert len(data) >= 1

    # Parse the results.
    data = data[0][sdss_id]
    psfmag = data["psfmag"]
    wavelengths = np.array(data["wavelengths"])
    flux = np.array(data["flux"])

    return sdss_id, wavelengths, flux, psfmag


def measure_hbeta(sdss_id, wavelengths, flux):
    # H-beta regions.
    region = [4847.875, 4876.625]
    lower = [4827.875, 4847.875]
    upper = [4876.625, 4891.625]

    # Extract line.
    inds = (wavelengths >= region[0]) * (wavelengths <= region[1])
    x = wavelengths[inds]
    y = flux[inds]

    # Build the linear continuum.
    il = (wavelengths >= lower[0]) * (wavelengths <= lower[1])
    y1 = simps(flux[il], x=wavelengths[il]) / (lower[1] - lower[0])
    x1 = np.mean(lower)

    iu = (wavelengths >= upper[0]) * (wavelengths <= upper[1])
    y2 = simps(flux[iu], x=wavelengths[iu]) / (upper[1] - upper[0])
    x2 = np.mean(upper)
    continuum = (y2 - y1) * (x - x1) / (x2 - x1) + y1

    # Integrate.
    integrand = 1 - y / continuum
    ew = simps(integrand, x=x)

    # Plot some diagnostics.
    pl.clf()

    pl.plot(x, y, "k")
    pl.plot(wavelengths[il], flux[il], "b")
    pl.plot(wavelengths[iu], flux[iu], "b")
    pl.plot(x, continuum, "--r")

    pl.xlabel(r"$\lambda$")
    pl.ylabel(r"$f$")
    pl.title(r"$\mathrm{{H}}\beta = {0:.2f}$".format(ew))

    pl.gca().xaxis.set_major_locator(MaxNLocator(6))
    pl.gca().yaxis.set_major_locator(MaxNLocator(6))

    pl.savefig("results/hbeta-{0}.pdf".format(sdss_id))

    return ew


def problem2():
    pl.figure(figsize=[6, 6])

    # Loop over the stars and save the equivalent widths.
    stars = [[181.65434, 33.977438],
             [180.92172, 25.519848],
             [180.26576, 33.729540],
             [178.76525, 31.807154],
             [181.53276, 34.094667],
             [177.30335, 30.758677],
             [180.73127, 33.741772],
             [174.63739, 30.887942]]

    ews, colors = [], []
    for ra, dec in stars:
        sdss_id, wavelengths, flux, psfmag = get_star(ra, dec)
        ew = measure_hbeta(sdss_id, wavelengths, flux)

        ews.append(ew)
        colors.append(psfmag["g"] - psfmag["r"])

    # Plot the "relation".
    pl.clf()
    pl.plot(colors, ews, "ok")
    pl.xlabel(r"$g - r$")
    pl.ylabel(r"$\mathrm{H}\beta$")
    pl.savefig("results/part2.pdf")


def problem3():
    pieces = [[1, 0.5], [6, 0.7], [10, 1.3], [60, 1.4], [90, 3.0]]

    # Sum over the piecewise linear components.
    s = 0.0
    for i in range(len(pieces) - 1):
        x1, y1 = pieces[i]
        x2, y2 = pieces[i + 1]
        a = (y2 - y1) / (x2 - x1)
        b = y1 - a * x1

        s += (a - 1.0) / 1.35 * (x2 ** (-1.35) - x1 ** (-1.35))
        s += b / 2.35 * (x2 ** (-2.35) - x1 ** (-2.35))

    return 0.144 * s


if __name__ == "__main__":
    problem2()
    print(problem3())
