# README for Delta Dockerfile

The Dockerfile is the blueprint for the delta image. It is built on top of an
NVIDIA cuda image to provide GPU support. It largely follows the instructions
here: https://delta-fusion.readthedocs.io/en/latest/notes/installing.html

## Using at NERSC

The image is called `registry.nersc.gov/das/delta:3.0`. 

As a backup, there is also an image that has delta installed inside: `registry.nersc.gov/das/delta-inside:1.0`


### Haswell

```
salloc -N 4 -C haswell -q interactive --image=registry.nersc.gov/das/delta:3.0
```

cd to wherever you have installed delta. For me it's 

```
cd /global/cscratch1/sd/stephey/delta/delta
```

And then launch delta:


```
OMP_NUM_THREADS=16 srun -n 16 -c 16 --cpu-bind=cores shifter python3 -m mpi4py.futures processor.py --config configs/hackathon_test.json --transport transport_tx  --num_ranks_preprocess=4 --num_ranks_analysis=12 --num_queue_threads=4 --run_id=test_25259_GT
```


### corigpu

On corigpu, request the image during your interactive job:

```
module load cgpu
salloc -N 1 -C gpu -G 1 -t 120 -A m499 --image=registry.nersc.gov/das/delta:3.0
```

```
OMP_NUM_THREADS=16 srun -n 16 -c 1 --cpu-bind=cores shifter python3 -m mpi4py.futures processor.py --config configs/hackathon_test.json --transport transport_tx  --num_ranks_preprocess=2 --num_ranks_analysis=2 --num_queue_threads=2 --run_id=test_25259_GT


### Debugging inside the container

```
shifter /bin/bash
```

### Known issues

Pytest has trouble running in the bind-mounted configuration for reasons I
don't understand. To run pytest you'll have to run in the container:

```
shifter --volume="/global/cscratch1/sd/stephey/delta/delta:/opt/delta" /bin/bash
pytest /opt/delta/tests
```
