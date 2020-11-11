# -*- Encoding: UTF-8 -*-


"""Defines infitnite-impulse bandpass filters."""

from scipy.signal import iirdesign, butter, sosfilt


def kernel_bandpass_sos(data, params):
    """Executes bandpass filtering.

    Uses `scipy.signal.sosfilt
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfilt.html>`_
    to filter a data sequence.

    Args:
        data (twod_chunk):
            Time-chunk of diagnostic data.
        params:
            Dictionary of arguments that are passed to `scipy.signal.sosfilt`.
            Second-order stable (sos) coefficients are stored in key `sos`.

    Returns:
        data (twod_chunk):
            Time-chunk with filtered data

    """
    y = sosfilt(params["sos"], data.data, axis=data.axis_t)
    data.data[:] = y[:]

    return(data)


class pre_bandpass_iir():
    """Implements bandpass filtering using scipy.signal iirdesign and sosfilt.

    Parameters passed to this class are forwarded to `scipy.signal.iirdesign
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirdesign.html>`_

    Please note that the pass and stop-band frequency ranges need to be
    provided in units of the Nyquist frequency. That is, to specify the pass band
    range from 5.1 to 19.9kHz for a signal with fnyq=250kHz, the parameter list for
    this callable needs to be given as
    :code:`{...'wp': [0.0204, 0.0796], ...}`.

    """

    def __init__(self, params):
        """Instantiates bandpass filter object.

        Args:
            params (dict):
                Will be passed to scipy.signal.iirdesign

        Returns:
            None
        """
        self.params = params
        self.sos = iirdesign(**params)

    def process(self, data_chunk, executor):
        """Bandpass-filters the time-chunk.

        Args:
            data_chunk (twod_chunk):
                Time-chunk of data
            executor (PEP-3148 executor):
                Executor on which call to submit will be executed

        Returns:
            data_chunk (twod_chunk):
                Time-chunk of data
        """
        # Do nothing and return the data
        params = {"sos": self.sos}
        fut = executor.submit(kernel_bandpass_sos, data_chunk, params)
        data_chunk = fut.result()
        return data_chunk


class pre_bandpass_fir():
    """Implements bandpass filtering using `scipy.signal.butter` and `sosfilt`.

    Parameters passed to this class are forwarded to `scipy.signal.butter
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html>`_

    Please note that the pass and stop-band frequency ranges need to be
    provided in units of the Nyquist frequency. That is, to specify the pass band
    range from 5 to 20kHz for a signal with fnyq=250kHz, the parameter list for
    this callable needs to be given as
    :code:`{...'Wn': [0.02, 0.08], ...}`.


    """
    def __init__(self, params):
        """Instantiates bandpass filter object.

        Args:
            params (dict):
                Will be passed to scipy.signal.iirdesign

        Returns:
            None
        """
        self.params = params
        # Remove unnamed arguments N and Wn before calling butter filter design with params
        N = params.pop("N")
        Wn = params.pop("Wn")
        self.sos = butter(N, Wn, **params)

    def process(self, data_chunk, executor):
        """Bandpass-filters the time-chunk.

        Args:
            data_chunk (twod_chunk):
                Time-chunk of data
            executor (PEP-3148 executor):
                Executor on which call to submit will be executed

        Returns:
            data_chunk (twod_chunk):
                Time-chunk of data
        """
        # Do nothing and return the data
        params = {"sos": self.sos}
        fut = executor.submit(kernel_bandpass_sos, data_chunk, params)
        data_chunk = fut.result()
        return data_chunk


# End of file pre_bandpass_iir.py
