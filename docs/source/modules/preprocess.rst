
Pre-processing
==============

The `processor` can be configured to pre-process data before submitting them
to the analysis pipeline. 

The pipeline is defined by the ``preprocess`` section in the configuration file.
Each entry in this section is interpreted as the name of a preprocessing routine
and its parameters.

For example, defining the entry


.. code-block:: json

    {...
        "preprocess":
        {
            "wavelet": {"wavelet": "db5", "method": "BayesShrink", "wavelet_levels": 5}
            "stft": {"nfft": 512, "fs": 5000000, "window": "hann", "overlap": 0.5, "detrend": linear}
        }
    ...}

Defines a pre-processing pipeline consisting of wavelet filtering followed by a short-time
Fourier transformation. The keys in the section, `wavelet` and `stft` in the example
here are mapped to preprocessor functions through
:py:meth:`preprocess.helpers.get_preprocess_routine` .

The example below shows how to set up a pre-processing pipeline

.. code-block:: python
    
    executor_pre = ThreadPoolExecutor(max_workers=4)
    my_preprocessor = preprocessor(executor_pre, cfg["preprocess"])

    while True:
        stream_data = reader.Get(stream_varname)
        msg = data_model_gen.new_chunk(stream_data, reader.CurrentStep())
        ...
        msg_pp = my_preprocessor.submit(msg)

First, an executor on which the pre-processing will be performed, is instantiated.
In the example here, this is a ThreadPoolExecutor that uses 4 worker threads on the
root node. Following that call, the pre-processing pipeline is instantiated. 
In the receiver loop, this pipeline is executed by calling 
:py:meth:`preprocess.preprocess.submit` with the just received
message. The pre-processed time-chunk is then avilable as `msg_pp`.


As a convention, all preprocessor classes define a method `process`, taking an
:class:`data_models.data_model` as input and returning the same object type. This
allows the routine to be stackable.
     


.. contents:: Contents 
    :local:

Preprocessing pipeline
----------------------
.. automodule:: preprocess.preprocess
    :members:

Short-Time Fourier Transformation
---------------------------------
.. automodule:: preprocess.pre_stft
    :members:

Wavelet Filtering
-----------------
.. automodule:: preprocess.pre_wavelet
    :members:

Bandpass Filtering
------------------
.. automodule:: preprocess.pre_bandpass
    :members:

Helper functions
----------------
.. automodule:: preprocess.helpers
    :members: