"""
Created on Jan 2023

@authors: Markus Bonse and Emily Garvin

We acknowledge the use and adaptation of the crosscorr function from PyAstronomy (https://pyastronomy.readthedocs.io/en/latest/)
"""


import numpy as np
from multiprocessing import Pool
from scipy import interpolate

from PyAstronomy.pyaC import pyaErrors as PE
from PyAstronomy.pyasl import _ic

# Functions we need in the CCF class -------------------------------------------


def crosscorr_rv_vec(w, f,
                     tw, tf,
                     rvmin, rvmax,
                     drv, mode="doppler",
                     skipedge=0,
                     edgeTapering=None):

    """
    Cross-correlate a spectrum with a template.


    The algorithm implemented here works as follows: For
    each RV shift to be considered, the wavelength axis
    of the template is shifted, either linearly or using
    a proper Doppler shift depending on the `mode`. The
    shifted template is then linearly interpolated at
    the wavelength points of the observation
    (spectrum) to calculate the cross-correlation function.

    Parameters
    ----------
    w : array
        The wavelength axis of the observation.
    f : array
        The flux axis of the observation.
    tw : array
        The wavelength axis of the template.
    tf : array
        The flux axis of the template.
    rvmin : float
        Minimum radial velocity for which to calculate
        the cross-correlation function [km/s].
    rvmax : float
        Maximum radial velocity for which to calculate
        the cross-correlation function [km/s].
    drv : float
        The width of the radial-velocity steps to be applied
        in the calculation of the cross-correlation
        function [km/s].
    mode : string, {lin, doppler}, optional
        The mode determines how the wavelength axis will be
        modified to represent a RV shift. If "lin" is specified,
        a mean wavelength shift will be calculated based on the
        mean wavelength of the observation. The wavelength axis
        will then be shifted by that amount. If "doppler" is
        specified (the default), the wavelength axis will
        properly be Doppler-shifted.
    skipedge : int, optional
        If larger zero, the specified number of bins will be
        skipped from the begin and end of the observation. This
        may be useful if the template does not provide sufficient
        coverage of the observation.
    edgeTapering : float or tuple of two floats
        If not None, the method will "taper off" the edges of the
        observed spectrum by multiplying with a sine function. If a float number
        is specified, this will define the width (in wavelength units)
        to be used for tapering on both sides. If different tapering
        widths shall be used, a tuple with two (positive) numbers
        must be given, specifying the width to be used on the low- and
        high wavelength end. If a nonzero 'skipedge' is given, it
        will be applied first. Edge tapering can help to avoid
        edge effects (see, e.g., Gullberg and Lindegren 2002, A&A 390).

    Returns
    -------
    dRV : array
        The RV axis of the cross-correlation function. The radial
        velocity refer to a shift of the template, i.e., positive
        values indicate that the template has been red-shifted and
        negative numbers indicate a blue-shift of the template.
        The numbers are given in km/s.
    CC : array
        The cross-correlation function.
    """

    if not _ic.check["scipy"]:
        raise (PE.PyARequiredImport(
            "This routine needs scipy (.interpolate.interp1d).",
            where="crosscorrRV",
            solution="Install scipy"))
    import scipy.interpolate as sci
    # Copy and cut wavelength and flux arrays
    w, f = w.copy(), f.copy()
    if skipedge > 0:
        w, f = w[skipedge:-skipedge], f[skipedge:-skipedge]

    if edgeTapering is not None:
        # Smooth the edges using a sine
        if isinstance(edgeTapering, float):
            edgeTapering = [edgeTapering, edgeTapering]
        if len(edgeTapering) != 2:
            raise (PE.PyAValError("'edgeTapering' must be a float or a "
                                  "list of two floats.",
                                  where="crosscorrRV"))
        if edgeTapering[0] < 0.0 or edgeTapering[1] < 0.0:
            raise (PE.PyAValError("'edgeTapering' must be (a) number(s) "
                                  ">= 0.0.",
                                  where="crosscorrRV"))

        # Carry out edge tapering (left edge)
        indi = np.where(w < w[0] + edgeTapering[0])[0]
        f[:, indi] *= np.sin((w[indi] - w[0]) / edgeTapering[0] * np.pi / 2.0)
        # Carry out edge tapering (right edge)
        indi = np.where(w > (w[-1] - edgeTapering[1]))[0]
        f[:, indi] *= np.sin((w[indi] - w[indi[0]]) / edgeTapering[
            1] * np.pi / 2.0 + np.pi / 2.0)

    # Speed of light in km/s
    c = 299792.458
    # Check order of rvmin and rvmax
    if rvmax <= rvmin:
        raise (PE.PyAValError("rvmin needs to be smaller than rvmax.",
                              where="crosscorrRV",
                              solution="Change the order of the parameters."))
    # Check whether template is large enough
    if mode == "lin":
        meanWl = np.mean(w)
        dwlmax = meanWl * (rvmax / c)
        dwlmin = meanWl * (rvmin / c)
        if (tw[0] + dwlmax) > w[0]:
            raise (PE.PyAValError(
                "The minimum wavelength is not covered by the template for all "
                "indicated RV shifts.",
                where="crosscorrRV",
                solution=["Provide a larger template", "Try to use skipedge"]))

        if (tw[-1] + dwlmin) < w[-1]:
            raise (PE.PyAValError(
                "The maximum wavelength is not covered by the template for all "
                "indicated RV shifts.",
                where="crosscorrRV",
                solution=["Provide a larger template", "Try to use skipedge"]))

    elif mode == "doppler":
        # Ensure that the template covers the entire observation for all shifts
        maxwl = tw[-1] * (1.0 + rvmin / c)
        minwl = tw[0] * (1.0 + rvmax / c)
        if minwl > w[0]:
            raise (PE.PyAValError(
                "The minimum wavelength is not covered by the template for "
                "all indicated RV shifts.",
                where="crosscorrRV",
                solution=["Provide a larger template", "Try to use skipedge"]))
        if maxwl < w[-1]:
            raise (PE.PyAValError(
                "The maximum wavelength is not covered by the template for "
                "all indicated RV shifts.",
                where="crosscorrRV",
                solution=["Provide a larger template", "Try to use skipedge"]))
    else:
        raise (PE.PyAValError("Unknown mode: " + str(mode),
                              where="crosscorrRV",
                              solution="See documentation for "
                                       "available modes."))

    # Calculate the cross correlation
    drvs = np.arange(rvmin, rvmax, drv)

    # cc = np.zeros(len(drvs))
    cc = np.zeros((int(f.shape[0]),
                   len(drvs)))

    for i, rv in enumerate(drvs):
        if mode == "lin":
            # Shift the template linearly
            fi = sci.interp1d(tw + meanWl * (rv / c), tf)

        elif mode == "doppler":
            # Apply the Doppler shift
            fi = sci.interp1d(tw * (1.0 + rv / c), tf)
        else:
            raise ValueError("mode not understood")

        # Shifted template evaluated at location of spectrum
        cc[:, i] = np.sum(f * fi(w), axis=1)

    return cc, drvs


def crosscorr_rv_vec_mp(dw, data_in, tw, tf,
                        rvmin, rvmax, drv,
                        mode, skipedge,
                        num_processes=32):
    pool = Pool(num_processes)

    parameters = list(zip([dw] * num_processes,
                          np.array_split(data_in, num_processes),
                          [tw] * num_processes,
                          [tf] * num_processes,
                          [rvmin] * num_processes,
                          [rvmax] * num_processes,
                          [drv] * num_processes,
                          [mode] * num_processes,
                          [skipedge] * num_processes))

    results = pool.starmap(crosscorr_rv_vec, parameters)
    pool.close()

    drvs = results[0][1]
    cc = np.concatenate([i[0] for i in results])

    return cc, drvs


def compute_template_norm(w, tw, tf,
                          rvmin,
                          rvmax,
                          drv,
                          mode="doppler",
                          skipedge=0):
    # Speed of light in km/s
    c = 299792.458

    w = w.copy()
    if skipedge > 0:
        w = w[skipedge:-skipedge]

    # Check whether template is large enough
    if mode == "lin":
        meanWl = np.mean(w)

    # Calculate the cross correlation
    drvs = np.arange(rvmin, rvmax, drv)
    template_term_norm = np.zeros(len(drvs))

    for i, rv in enumerate(drvs):

        if mode == "lin":
            # Shift the template linearly
            fi = interpolate.interp1d(tw + meanWl * (rv / c), tf)

        elif mode == "doppler":
            # Apply the Doppler shift
            fi = interpolate.interp1d(tw * (1.0 + rv / c), tf)
        else:
            raise ValueError("mode not understood")

        # Shifted template evaluated at location of spectrum
        template_term_norm[i] = np.sum(fi(w) ** 2)

    return template_term_norm, drvs


# The CCF class ----------------------------------------------------------------

class CCFSignalNoise:

    def __init__(self, signal, noise,
                 dw, tw, tf,
                 rvmin, rvmax, drv,
                 mode="doppler",
                 skipedge=0,
                 num_processes=32):
        # Save the input values
        self.m_dw = dw
        self.m_tw = tw
        self.m_tf = tf
        self.m_rvmin = rvmin
        self.m_rvmax = rvmax
        self.m_drv = drv
        self.m_mode = mode
        self.m_skipedge = skipedge
        self.m_noise = noise
        self.m_signal = signal
        self.m_num_processes = num_processes

        # 1.) cross correlate signal and noise
        self.m_cc_signal, self.m_drvs = self._cross_correlate(signal)
        self.m_cc_noise, _ = self._cross_correlate(noise)

        # 2.) compute the template
        self.m_template_term_norm, _ = compute_template_norm(self.m_dw,
                                                             self.m_tw,
                                                             self.m_tf,
                                                             self.m_rvmin,
                                                             self.m_rvmax,
                                                             self.m_drv,
                                                             self.m_mode,
                                                             self.m_skipedge)

    def _cross_correlate(self, data_in):
        return crosscorr_rv_vec_mp(self.m_dw, data_in,
                                   self.m_tw, self.m_tf,
                                   rvmin=self.m_rvmin,
                                   rvmax=self.m_rvmax,
                                   drv=self.m_drv,
                                   mode=self.m_mode,
                                   skipedge=self.m_skipedge,
                                   num_processes=self.m_num_processes)

    def get_data(self, alpha):
        cc_signal_and_noise = self.m_cc_noise + self.m_cc_signal * alpha
        signal_and_noise = self.m_noise + self.m_signal * alpha
        signal_noise_norm = np.sum(signal_and_noise ** 2, axis=1)
        normalization = np.matmul(signal_noise_norm[:, np.newaxis],
                                  self.m_template_term_norm[:, np.newaxis].T)

        normalization = np.sqrt(normalization)

        return cc_signal_and_noise / normalization, self.m_drvs
