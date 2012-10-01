#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as pl
import requests


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
    assert len(data) == 1

    return data[0]


def get_star(ra, dec, radius=0.005):
    # Find the SDSS ID.
    sdss_id = find_star(ra, dec, radius)

    # Build request.
    url = sdss_url + "/spectrum"
    pars = {"id": sdss_id, "format": "json", "fields": "flux,wavelengths"}

    # Submit request.
    r = requests.get(url, params=pars)
    if r.status_code != requests.codes.ok:
        r.raise_for_status()

    # Get the contents of the response.
    data = r.json
    assert len(data) == 1

    data = data[0][sdss_id]

    wavelengths = np.array(data["wavelengths"])
    flux = np.array(data["flux"])

    pl.plot(wavelengths, flux, "k")
    pl.savefig(sdss_id + ".png")


if __name__ == "__main__":
    get_star(181.65434, 33.977438)
