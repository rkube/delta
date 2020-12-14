Welcome to delta-fusion's documentation!
========================================

`Delta`(the aDaptive near rEaL-Time Analysis framework) facilitates near real-time streaming
analysis of big fusion data on remote HPC resources. It implements software components
for loading and staging measurement data, streaming, analysis, and storage. `Delta` combines these
components into a framework for adaptive near-real time analysis of streaming data, targeted 
towards use-cases in fusion energy research. It consists of multiple executables that send,
receive, and process data on different machines. The picture below gives an 
overview of Delta.




.. toctree::
   :maxdepth: 2
   :caption: Contents: 


.. figure:: delta_arch_v02.png
    :align: center

    Target architecture of Delta 

At the data generation site, a `generator` reads measurement data from file and stages
it for streaming. The data stream is received by a `middleman` that forwards it 
to the `processor`, which executes data analysis kernels on a supercomputer.
The analysis results are stored and made available to web-clients by a separate
webserver.

This documentation describes the capabilities of `Delta`, how to configure it and
how to launch distributed anaylsis workflows.


Running Delta
=============


.. toctree::
   :maxdepth: 2
   :caption: Running Delta

   notes/installing
   notes/configuring
   notes/launching


Package reference
=================

.. toctree::
    :maxdepth: 1
    :caption: Package reference

    modules/data_models
    modules/sources
    modules/streaming
    modules/preprocess
    modules/analysis
    modules/storage
    modules/configuration




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

