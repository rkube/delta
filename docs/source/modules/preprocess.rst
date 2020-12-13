
Pre-processing
==============

This module provides the building blocks to construct a composable pre-processing pipeline.
Data chunks that are passed into the pre-processing pipeline are sequentially transformed by
each pre-processing routine. The pipeline is defined by the ``preprocess`` section in the
configuration file. Each entry in this section is interpreted as the name of a preprocessing routine
and its parameters. The routines itself are executed on a PEP-3148 style executor.

For example, the entry


.. code-block:: json

    "preprocess" 
    {
      "pre1": {"parameter1": 1, "parameter2": 2},
      "pre2": {"parameter1": 1, "parameter2": "val2"}
    }


defines a pre-processing pipeline consisting of two routines. The keys `pre1` and `pre2` 
in the `preprocess` section are mapped to preprocessor classes by
:py:meth:`preprocess.helpers.get_preprocess_routine`. The preprocessor classes itself expose a 
`process` member, which takes a data chunk as input and returns the transformed data chunk.
This function serves as a wrapper around a function call.
The parameter dictionaries in the configuration file above are stored by the preprocessor 
class instances and are passed to the wrapped function call.

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


As a common interface, all preprocessor classes define the method `process`, taking an
:class:`data_models.data_model` as input and returning the same object type. This
allows to compose preprocessing routines into a pipeline.




.. contents:: Contents 
    :local:


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

Helper functions
----------------
.. automodule:: preprocess.helpers

