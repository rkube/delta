# DELTA-FUSION (aDaptive rEaL Time Analysis of big fusion data) – Ray-based version

In this documentation, we show the changes that have been made to DELTA to use Ray distributed execution framework instead of mpi4py. For more information about Ray, see [Ray](https://www.ray.io/).  Ray is a fast, simple distributed execution framework. It provides a simple, universal API for building distributed applications. Ray enables us to run scripts in a heterogeneous (CPUs/GPUs) environment.  It also makes it easy to scale applications and leverage state-of-the-art machine learning libraries. With Ray, we can easily parallelize machine learning tasks with simple changes, which is one of this reform's main goals.

## What have been changed:
Most of the changes were done on the processor.py and its supplementary files (DELTA classes). Here we describe the main changes we made on DELTA. Ray was first initiated using this code line:  
```python
ray.init(address=os.environ["ip_head"],ignore_reinit_error=True) 
```
this way we ignore initiating Ray in other DELTA classes (libraries). Three core changes were then done as follows: 

### 1. Read the data and push them into a Queue: 
The first step in DELTA is to read data files and set them as chunks into a queue. Reader was initiated using reader_gen from streaming.reader_nompi file. Notice here, we used the nompi version. This file streaming.reader_nompi file was not changed, it was used as is in the old DELTA version. Reader then reads the data chunks and put them into a Ray queue. Queue library was imported from ray.util.queue. 

### 2. Start the workers (Ray tasks):
Each worker is designed to run pre-processing and analysis tasks on a data chunk. Once it finishes this with its chunk, he pulls another chunk from the queue until it the queue gets empty. All workers start in parallel – and each worker is doing all its job on one node. Therefore, we initialized the number of workers to be the number of the available nodes -1, since one node is reserved to be the head node. The following is the line code we use to start the workers, 

```python
num_workers = int(os.environ["SLURM_NTASKS"]) - 1
workers = [consume.remote(q,my_preprocessor,my_task_list,j) for j in range(num_workers)]
```

### 3. Pre-Processing tasks:

Next, each worker using the resources of its node will run the pre-processing tasks. The following scripts were modified: ![image](https://user-

```bash
Helpers.py, pre_bandpass.py, pre_plot.py, pre_stft.py, pre_wavelet.py, preprocess.py
```
The takes that have been changed to use Ray are bandpass filtering, wavelet filtering, short-time Fourier transformation. \\
When these tasked executed, it will use CPUs that are available in the node. 
Notes: To update the data chunk after it’s pre-processed. Data_models.base_models.py was modified and a new method (function) was added named: 
```python
update_data(self,new_data)
```
to make the data writable, since Ray changes its permission to be unwritable. 


### 4. Analysis tasks:
Once a worker finishes the preprocessing, it takes the pre-processed data chunk and applies the analysis task. One type of these tasks uses only CPUs, other tasks use GPUs. 
CPUs analysis tasks:
```bash
task_crosscorr, task_coherence, task_crossphase, task_crosspower
```
GPUs analysis tasks:
```bash
task_crosscorr_cu, task_coherence_cu, task_crossphase_cu, task_crosspower_cu 
```






