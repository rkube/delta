Welcome to delta-fusion's documentation!
========================================

`Delta` (the a **D** aptive near r **E** a **L** - **T** ime **A** nalysis framework) facilitates near real-time
streaming and analysis of big fusion data on remote HPC resources. It implements software components
for loading and staging measurement data, streaming, analysis, and storage. `Delta` combines these
components into a framework for adaptive near-real time analysis of streaming data, targeted 
towards use-cases in fusion energy research. It consists of multiple executables that send,
receive, and process data on different machines. The picture below gives an overview of Delta.


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

Running Delta
=============
Describes how to install, configure, and launch distributed analysis workflows.


.. toctree::
   :maxdepth: 2
   :caption: Running Delta

   notes/installing
   notes/configuring
   notes/launching


Package reference
=================
Documentation of individual software modules

.. toctree::
    :maxdepth: 1
    :caption: Package reference

    modules/sources
    modules/data_models
    modules/streaming
    modules/preprocess
    modules/analysis
    modules/storage


Presentations and Papers
========================
Delta has been presented at multiple scientific conferences.

SciPy 2020
^^^^^^^^^^

* `Proceedings paper <http://conference.scipy.org/proceedings/scipy2020/ralph_kube.html>`_
* `Presentation <https://www.youtube.com/watch?v=8udbAD-KZdA&t=4s>`_


23rd Conference on High Temperature Plasma Diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* `Presentation <https://www.youtube.com/watch?v=56d93cN9oNo&t=370s>`_

SC 2020 Workshop
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
2020 IEEE/ACM HPC for Urgent Decision Making Workshop

* `Paper <https://conferences.computer.org/scwpub/#!/toc/23?>`_



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

