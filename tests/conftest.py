# -*- Encoding: UTF-8 -*-

import pytest
import json


@pytest.fixture(scope="module")
def config_all():
    """Provides a configuration object for all unit tests."""    
    config_str = """{
        "diagnostic":
          {
    "name": "kstarecei",
    "shotnr": 18431,
    "datasource":
    {
      "source_file": "/global/cscratch1/sd/rkube/KSTAR/kstar_streaming/022289/ECEI.022289.GT.h5",
      "chunk_size": 10000,
      "num_chunks": 500,
      "channel_range": ["L0101-2408"],
      "datatype": "float"
    }
  },
  "transport_nersc":
  {
    "datapath": "tmp_nersc",
    "engine": "BP4",
    "params":
    {
      "IPAddress": "128.55.205.18",
      "Timeout": "120",
      "Port": "50001",
      "TransportMode": "fast"
    }
  },
  "storage":
  {
    "backend": "numpy",
    "basedir": "/global/homes/r/rkube/tmp"
  },
 "preprocess": 
{
   "bandpass_iir": {"wp": [0.02, 0.036], "ws": [0.0192, 0.0368], "gpass": 1, "gstop": 40, "ftype": "ellip"},
   "stft": {"nfft": 512, "fs": 500000, "window": "hann", "overlap": 0.5, "noverlap": 256, "detrend": "constant", "full": true}
   },
  "analysis": {
    "crossphase": {
      "channel_chunk_size": 32768,
      "ref_channels": [1, 1, 24, 8],
      "cmp_channels": [1, 1, 24, 8]
    },
    "crosspower": {
      "channel_chunk_size": 32768, 
      "ref_channels": [1, 1, 24, 8], 
      "cmp_channels": [1, 1, 24, 8]
    },
    "crossphase": {
      "channel_chunk_size": 32768, 
      "ref_channels": [1, 1, 24, 8],
      "cmp_channels": [1, 1, 24, 8]
    },
    "coherence": {
      "channel_chunk_size": 32768,
      "ref_channels": [1, 1, 24, 8],
      "cmp_channels": [1, 1, 24, 8]
    }
  }
  }"""

    config = json.loads(config_str)
    return config



@pytest.fixture(scope="module")
def config_analysis_cy():
    """Provides a configuration object for all unit tests."""    
    config_str = """{
        "diagnostic":
          {
    "name": "kstarecei",
    "shotnr": 18431,
    "datasource":
    {
      "source_file": "/global/cscratch1/sd/rkube/KSTAR/kstar_streaming/022289/ECEI.022289.GT.h5",
      "chunk_size": 10000,
      "num_chunks": 500,
      "channel_range": ["L0101-2408"],
      "datatype": "float"
    }
  },
  "transport_nersc":
  {
    "datapath": "tmp_nersc",
    "engine": "BP4",
    "params":
    {
      "IPAddress": "128.55.205.18",
      "Timeout": "120",
      "Port": "50001",
      "TransportMode": "fast"
    }
  },
  "storage":
  {
    "backend": "numpy",
    "basedir": "/global/homes/r/rkube/tmp"
  },
 "preprocess":
{
   "bandpass_iir": {"wp": [0.02, 0.036], "ws": [0.0192, 0.0368], "gpass": 1, "gstop": 40, "ftype": "ellip"},
   "stft": {"nfft": 512, "fs": 500000, "window": "hann", "overlap": 0.5, "noverlap": 256, "detrend": "constant", "full": true}
   },
  "analysis": {
    "crosspower_cy": {
      "channel_chunk_size": 32768, 
      "ref_channels": [1, 1, 24, 8], 
      "cmp_channels": [1, 1, 24, 8]
    },
    "crossphase_cy": {
      "channel_chunk_size": 32768, 
      "ref_channels": [1, 1, 24, 8],
      "cmp_channels": [1, 1, 24, 8]
    },
    "coherence_cy": {
      "channel_chunk_size": 32768,
      "ref_channels": [1, 1, 24, 8],
      "cmp_channels": [1, 1, 24, 8]
    }
  }
  }"""

    config = json.loads(config_str)
    return config



@pytest.fixture(scope="module")
def config_analysis_cu():
    """Provides a configuration object for all unit tests."""    
    config_str = """{
        "diagnostic":
          {
    "name": "kstarecei",
    "shotnr": 18431,
    "datasource":
    {
      "source_file": "/global/cscratch1/sd/rkube/KSTAR/kstar_streaming/022289/ECEI.022289.GT.h5",
      "chunk_size": 10000,
      "num_chunks": 500,
      "channel_range": ["L0101-2408"],
      "datatype": "float"
    }
  },
  "transport_nersc":
  {
    "datapath": "tmp_nersc",
    "engine": "BP4",
    "params":
    {
      "IPAddress": "128.55.205.18",
      "Timeout": "120",
      "Port": "50001",
      "TransportMode": "fast"
    }
  },
  "storage":
  {
    "backend": "numpy",
    "basedir": "/global/homes/r/rkube/tmp"
  },
 "preprocess":
{
   "bandpass_iir": {"wp": [0.02, 0.036], "ws": [0.0192, 0.0368], "gpass": 1, "gstop": 40, "ftype": "ellip"},
   "stft": {"nfft": 512, "fs": 500000, "window": "hann", "overlap": 0.5, "noverlap": 256, "detrend": "constant", "full": true}
   },
  "analysis": {
    "crossphase_cu": {
      "channel_chunk_size": 32768,
      "ref_channels": [1, 1, 24, 8],
      "cmp_channels": [1, 1, 24, 8]
    }
  }
  }"""

    config = json.loads(config_str)
    return config


@pytest.fixture()
def stream_attrs_022289():
    """Dummy substitute for ECEI parameters typically read from HDF5.

    These parameters are taken from shot 022289
    """
    stream_attrs = {"dev": "GT",
                    "TriggerTime": [-0.1, 61.1, 60],
                    "t_norm": [-0.119, -0.109],
                    "SampleRate": 500000,
                    "TFcurrent": 18000.0,
                    "Mode": "X",
                    "LoFreq": 79.5,
                    "LensFocus": 503,
                    "LensZoom": 200}

    return stream_attrs


@pytest.fixture()
def stream_attrs_018431():
    """Dummy substitute for ECEI parameters typically read from HDF5.

    These parameters are taken from shot 018431
    """
    stream_attrs = {"dev": "L",
                    "TriggerTime": [-0.12, 61.12, 60],
                    "t_norm": [-0.119, -0.109],
                    "SampleRate": 500000,
                    "TFcurrent": 23000.0,
                    "Mode": "O",
                    "LoFreq": 81,
                    "LensFocus": 80,
                    "LensZoom": 340}

    return stream_attrs




@pytest.fixture(scope="module")
def gen_sine_waves():
    """Generate sine wave data for coherence kernel.

    Creates two signals with two frequencies.
    Each frequency has a distinct phase shift.

    See kstar_test_coherence.ipynb
    """
    import numpy as np
    from scipy.signal import stft

    # Number of realizations of the signal
    num_realizations = 5
    # Sampels per realization
    samples_per_realization = 100
    t0 = 0.0
    t1 = 1.0
    dt = (t1 - t0) / samples_per_realization
    # Time range of a single realization
    trg = np.arange(t0, t1 * num_realizations, dt)

    # Pre-allocate wave data array
    wave_data = np.zeros([2, num_realizations * samples_per_realization])

    # Base frequencies and phase shift of each frequency
    f0 = 2.0
    f1 = 8.0
    delta_phi_f0 = 0.25 * (t1 - t0)
    delta_phi_f1 = 0.5 * (t1 - t0)

    # Calculate y
    wave_data[0, :] = np.sin(2.0 * np.pi * f0 * trg) +\
        np.sin(2.0 * np.pi * f1 * trg)
    wave_data[1, :] = np.sin(2.0 * np.pi * f0 * trg + delta_phi_f0) +\
        np.sin(2.0 * np.pi * f1 * trg + delta_phi_f1)

    # Pre-allocate FFT data array.
    num_bins = 6
    nfft = samples_per_realization // 2 + 1
    fft_data = np.zeros([2, nfft, num_bins], dtype=np.complex128)
    window = np.ones(samples_per_realization) / samples_per_realization
    for ch in [0, 1]:
        f_s = stft(wave_data[ch, :], nperseg=samples_per_realization, noverlap=0,
                   window=window)
        fft_data[ch, :, :] = f_s[2][:, ]

    return fft_data

# End of file conftest.py
