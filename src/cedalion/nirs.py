import numpy as np
import xarray as xr
from numpy.typing import ArrayLike
from scipy.interpolate import interp1d

import cedalion.validators as validators
import cedalion.xrutils as xrutils
from importlib import resources


def get_extinction_coefficients(spectrum: str, wavelengths: ArrayLike):
    """Provide a matrix of extinction coefficients from tabulated data."""
    if spectrum == "prahl":
        with resources.open_text(
            "cedalion.data", "prahl_absorption_spectrum.tsv"
        ) as fin:
            coeffs = np.loadtxt(fin, comments="#")

        chromophores = ["HbO", "HbR"]
        spectra = [
            interp1d(coeffs[:, 0], np.log(10) * coeffs[:, i] / 10) for i in [1, 2]
        ]  # convert units from cm^-1/ M to mm^-1 / M

        E = np.array([spec(wl) for spec in spectra for wl in wavelengths]).reshape(
            len(spectra), len(wavelengths)
        )

        E = xr.DataArray(
            E,
            dims=["chromo", "wavelength"],
            coords={"chromo": chromophores, "wavelength": wavelengths},
            attrs={"units": "mm^-1 / M"},
        )
        E = E.pint.quantify()
        return E
    else:
        raise ValueError(f"unsupported spectrum '{spectrum}'")


def channel_distances(amplitudes: xr.DataArray, geo3d: xr.DataArray):
    """Calculate distances between channels."""
    validators.has_channel(amplitudes)
    validators.has_positions(geo3d, npos=3)
    validators.is_quantified(geo3d)

    diff = geo3d.loc[amplitudes.source] - geo3d.loc[amplitudes.detector]
    dists = xrutils.norm(diff, "pos")
    dists = dists.rename("dists")

    return dists


def beer_lambert(
    amplitudes: xr.DataArray,
    geo3d: xr.DataArray,
    dpf: xr.DataArray,
    spectrum: str = "prahl",
    calc_od: bool = True
):
    """Calculate concentration changes from amplitude data."""
    validators.has_channel(amplitudes)
    validators.has_wavelengths(amplitudes)
    validators.has_wavelengths(dpf)
    validators.has_positions(geo3d, npos=3)

    E = get_extinction_coefficients(spectrum, amplitudes.wavelength)

    Einv = xrutils.pinv(E)

    dists = channel_distances(amplitudes, geo3d)
    dists = dists.pint.to("mm")

    if calc_od:
        optical_density = -np.log(amplitudes / amplitudes.mean("time"))
    else:
        optical_density = amplitudes
    # conc = Einv @ (optical_density / ( dists * dpf))
    conc = xr.dot(Einv, optical_density / (dists * dpf), dims=["wavelength"])
    conc = conc.pint.to("micromolar")
    conc = conc.rename("concentrations")

    return conc
