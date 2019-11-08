# Coding: UTF-8 -*-


def power_spectrum(data, **kwargs):
    """Implements an overlapped segmented averaging of modified periodograms.
    Currently scipy.signal.welch

    See
    * Originial implementation in kstar fluctana

    * Discussion in
      'Spectrum and spectral density estimation by the Discrete Fourier transform (DFT),
      including a comprehensive list of window functions and some new flat-top windows'
      G. Heinzel, A. Rudiger and R. Schilling (2002)

    Input:
    ======
    data : channel data to be analyzed
    **kwargs : keyword arguments to be passed into wrapped function. See documentation of wrapped function.
    
    
    Returns:
    ========
    f   : ndarray, vector of frequency bins
    Pxx : ndarray, vector of power spectral densities
    """
    
    from scipy.signal import welch
    f, Pxx = welch(data, **kwargs)

    return(f, Pxx)


# End of file analysis.py