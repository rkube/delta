# DELTA-FUSION (aDaptive rEaL Time Analysis of big fusion data) – Ray-based version

In this documentation, we show the changes that have been made to DELTA to use Ray distributed execution framework instead of mpi4py. For more information about Ray, see [Ray](https://www.ray.io/).  Ray is a fast, simple distributed execution framework. It provides a simple, universal API for building distributed applications. Ray enables us to run scripts in a heterogeneous (CPUs/GPUs) environment.  It also makes it easy to scale applications and leverage state-of-the-art machine learning libraries. With Ray, we can easily parallelize machine learning tasks with simple changes, which is one of this reform's main goals.



## What have been changed:

Most of the changes were done on the processor.py and its supplementary files (DELTA classes). Here we describe the main changes we made on DELTA. Ray was first initiated using this code line: 
```python
ray.init(address=os.environ["ip_head"],ignore_reinit_error=True)
```
this way we ignore initiating Ray in other DELTA classes (libraries). Three core changes were then done as follows: 

