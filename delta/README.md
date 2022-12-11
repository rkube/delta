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

***
Notes

* Cython tasks will not run on this version of DELTA. An error will be raised if there is a Cython task required in the configuration file. 
* Files that were changed are:
  ```bash
  helpers.py, task_base.py, task_spectral.py, task_spectral_cu.py. 
  ``` 
  Other files were not changed. Kernels also were not changed. 
  In task_base.py: this includes the base class for each analysis task, one GPU (besides 1 CPU that is reserved by default) was reserved to run the main function calc_and_store() that executes the analysis kernel. If you would like to specify more GPUs and CPUs change the decorator on the top of the function definition
  ```python
  @ray.remote(num_gpus=1)
  ```
  but you should be aware that each time there should be enough resources for the worker to use, otherwise, not all chunks will be executed. This was one of the main problems we faced when DELTA was changed to use Ray.  

#### These are the changed that were made, lets talk now about how to run DELTA Ray-based version 


\\
\\
\\
\\
\\
\\
\\
\\


## Running DELTA Ray-based version: 

DELTA Ray version is designed to be run on Perlmutter. 


<img width="335" alt="image" src="https://user-images.githubusercontent.com/48891624/183760124-6c31cb4d-6403-4a70-8d94-dc4ad7321d1c.png">
Figure: this figure shows the programming architecture of DELTA Ray-based version. We start by initiating the head and the worker nodes using a Slurm with a bash script. The head node works as an organizer node, reads the data from the resources using ADIOS, and fills the queue with data chunks. Then, each worker pulls out a data chunk and applies the pre-processing tasks and then the analysis tasks. Each worker utilizes one node, and each task is executed using multiple CPUs or GPUs, depending on the type of the task. 



### 1. Setting up the configuration file: 
The configuration file inside the configs directory that is used is called: hackathon_test.json. In this file, we specify all the required parameters/directories and tasks to be used when running DELTA. 
Under “storage”, add the following to store the results at the end: 
```json
"backend": "numpy",
"basedir":"/pscratch/sd/m/maburidi/data_storage/data_storage_ray_delta” 
```
This is the directory where to store the data after analyzing it, change it to point to your directory. You can choose “backend” to be “mongodb”. 

In the current config file, there are two pre-processing tasks: bandpass_fir and stft. There are also: no_wavelet and no_plot: those to prevent other tasks to be run.  
For now, and for comparison reason we just added two analysis tasks, those are: crosscorr and crosspower_cu. There are also: crosspower_cy, crossphase_cy and coherence_cy, those are Cython based tasks, and it will not run since this version is not designed to run those tasks. A warning will be raised once passed. 


### 2. Initiating the Head and the worker nodes:

The first thing is that we need to write and run a bash script to initiate the head and the worker nodes. This bash script is included in the repository and called: 
```bash
slurm-delta.sh
```
Notes

* Delta will run inside DELTA shifter image.
* After running many experiments, we concluded that the following flags in the bash scripts may be chosen like this for optimal performance: 
   ```bash
   #SBATCH --ntasks-per-node=1          
   #SBATCH --gpus-per-task=4               
   #SBATCH --cpus-per-task=32    
   ```
   Notes: if these flags were not set correctly, we noticed that not all analysis tasks will be applied, so not all the data chunks would be analyzed    correctly.  
   
* SBATCH --nodes=4.   This will determine the number of workers, here it’s three.
* To run. DELTA: inside the terminal, run:  sbatch slurm-delta.sh 



## Testing the results: 
To test the results, we compare the resulted new data chunks of DELTA Ray-based version with DELTA-mpi version. To test, be in tests directory and run:
   ```bash
shifter --image=registry.nersc.gov/das/delta:5.0 pytest -s test_ray.py --dir_1=/pscratch/sd/m/maburidi/data_storage/data_storage_ray_delta/ --dir_2=/pscratch/sd/m/maburidi/data_storage/data_storage_old_delta/ --task_name=task_crosscorr --num_chunks=100
   ```
  required arguments: 

   ```bash
    --dir_1:   directory where DELTA-Ray analysis results are stored
    --dir_2:   directory where DELTA-mpi analysis results are stored
    --task_name:  analysis task name ex. task_crosscorr, task_crosscorr_cu  
    --num_chunks: number of chunks to be compared 
   ```
