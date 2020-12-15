
Streaming
=========

`Delta` moves data between generator, middleman, and processor using the interface defined
in this module. Delta employs the `ADIOS2 <https://adios2.readthedocs.io>`_ library for data streaming.

Example code to instantiate a writer and stream data would look like this:

.. code-block:: python
    
    dataloader = get_loader(cfg)
    writer = writer_gen(cfg["transport"], gen_channel_name(cfg["diagnostic"])
    wrter.DefineVariable(gen_var_name(cfg), dataloader.get_chunk_shape(), dataloader.dtype)
    writer.Open()
    writer.DefineAttributes("stream_attrs", {"param1": 1.0, "param2": 2.0})

    for nstep, chunk in enumerate(dataloader.batch_generator()):
        writer.BeginStep()
        writer.put_data(chunk)
        writer.EndStep()
    
    writer.Close()


During instantiation, the writer uses `cfg['transport']` to set appropriate ADIOS2 parameters,
such as the `streaming protocol <https://adios2.readthedocs.io/en/latest/engines/engines.html>`_ ,
and additional parameters. For example, the configuration section

.. code-block::

    "transport":
    {
        "engine": "BP4",
        "params":
        {
            "Threads: 4",
            "InitialBufferSize": "1Gb"
        }
    }

will instantiate :py:class:`streaming.reader_mpi.reader_gen` that expects input
from a BP4-file. Details on BP4 and valid parameters can be found in the
`ADIOS2 documentation <https://adios2.readthedocs.io/en/latest/engines/engines.html#bp4>`_ .

Since :py:class:`streaming.writers.writer_gen` and :py:class:`streaming.reader_mpi.reader_gen` 
need to open the same channel name, the convenience function 
:py:func:`data_models.helpers.gen_channel_name` generates such a name from the `diagnostic`
object in the configuration file.

To send data through the stream, a 
`variable <https://adios2.readthedocs.io/en/latest/components/components.html?highlight=definevariable#variable>`_ 
has to be defined with a shape and an expected datatype.
After the stream has been opened, one can define 
`attributes <https://adios2.readthedocs.io/en/latest/components/components.html?highlight=attribute#attribute>`_ 
on it. Data is written to the stream in a step-based manner. After writing has finished, the writer needs to be closed.


Example code to receive the data stream would look like this:

.. code-block:: python

    reader = reader_gen(cfg["transport"], gen_channel_name(cfg["diagnostic"])
    reader.Open()
    stream_attrs = None
    stream_varname = gen_var_name(cfg)

    while True:
        stepStatus = reader.BeginStep(timeoutSeconds=1.0)
        if stepStatus:
            if stream_attr is is None:
                stream_attrs = reader.get_attrs("stream_attrs")
            
            stream_data = reader.Get(stream_varname)
        else:
            break

Like the writer, the reader uses cfg['transport'[ during instantiation to set appropriate 
ADIOS2 parameters. The stream name to receive is defined by `gen_channel_name` helper function.
Attributes are initialized empty and the variable name to receive on the stream is defined by 
the `gen_var_name` helper function. Entering the receive loop, stepStatus indicates whether a 
new step is available. If so, at first the attributes of the stream are inquired, if this has
not been done before. Then a data packet is received. If no new steps are received after waiting 
for the defined timeout, it is assumed that the sender has stopped transmitting and receive is 
aborted.


.. contents:: Contents
    :local:

reader_mpi
----------

.. automodule:: streaming.reader_mpi
    :members:
    :special-members: __init__


writers
-------

.. automodule:: streaming.writers
    :members:
    :special-members: __init__

stream_stats
------------

.. automodule:: streaming.stream_stats
    :members:
    :special-members: __init__
