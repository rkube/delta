
Pre-processing
==============

This module provides the building blocks to construct a composable pre-processing pipeline.
Data chunks that are passed into the pre-processing pipeline are sequentially transformed by
each pre-processing routine. The pipeline is defined by the ``preprocess`` section in the
configuration file. Each entry in this section is interpreted as the name of a preprocessing routine
and its parameters. The routines itself are executed on a PEP-3148 style executor.

For example, the entry

.. code-block:: 

    "preprocess": 
    {
      "bandpass_fir": {"N": 5, "Wn": [0.02, 0.036], "btype": "bandpass", "output": "sos"},
      "plot": {"time_range": [2.7175, 2.7178], "plot_dir": "/home/user/delta_run/plots/"}
    }


defines a pre-processing pipeline consisting of two routines,
:py:class:`preprocess.pre_bandpass.bandpass_fir` and 
:py:class:`preprocess.pre_plot.pre_plot`. The function :py:func:`preprocess.helpers.get_preprocess_routine`
implements a mapping from keys such as :code:`bandpass_fir` and
:code:`plot` to the respective preprocessor classes. 

Preprocessor classes are instantiated like this:

.. code-block:: python

    pp_fir = pre_bandpass_fir(params)
    pp_plot = pre_plot(params)

The parameters are stored as member variables by the class objects. To preprocess time chunks,
preprocessing classes define  :py:meth:`preprocess.wavelet.process` member. This preprocesses a
time-chunk on an executor and returns the transformed time-chunk.

By returning the same data type as was passed into the `process` member, preprocessing 
routines can be combined into a preprocessing pipeline:

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
:py:meth:`preprocess.preprocessor.submit` with the just received
message. The pre-processed time-chunk is then avilable as `msg_pp`.


As a common interface, all preprocessor classes define the method `process`, taking an
:class:`data_models.data_model` as input and returning the same object type. This
allows to compose preprocessing routines into a pipeline.




.. contents:: Contents 
    :local:

preprocess
----------

.. automodule:: preprocess.preprocess
    :members:
    :special-members: __init__


Short-Time Fourier Transformation
---------------------------------
.. automodule:: preprocess.pre_stft
    :members:
    :special-members: __init__

Wavelet Filtering
-----------------
.. automodule:: preprocess.pre_wavelet
    :members:
    :special-members: __init__

Bandpass Filtering
------------------
.. automodule:: preprocess.pre_bandpass
    :members:
    :special-members: __init__

Plotting
--------
.. automodule:: preprocess.pre_plot
    :members:
    :special-members: __init__

Preprocessing Helper functions
------------------------------
.. autofunction:: preprocess.helpers.get_preprocess_routine


