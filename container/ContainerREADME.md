# README for Delta Containers

The Dockerfile is the blueprint for the Delta image. It is built on top of an
NVIDIA CUDA 11.5 image to provide GPU support. It largely follows the instructions
here: https://delta-fusion.readthedocs.io/en/latest/notes/installing.html

## Building, Pushing, and Pulling

To build the Delta image, clone this repo and cd into the top level directory.

Note that since we're using `ADD` to copy in the Delta `requirements.txt`
file, you'll need to do the build from the directory where it is located.

Here we are building on a NERSC staff podman node:

```
podman build -t registry.nersc.gov/das/delta:4.0 . -f container/delta-outside/Dockerfile
```

We're using `registry.nersc.gov` to store our image. You may first need to log in
via:

```
podman login registry.nersc.gov
```

Note your NERSC username and password (no OTP) should be used.

To push:

```
podman push registry.nersc.gov/das/delta:4.0
```

To pull onto Cori/Perlmutter:

```
shifterimg login registry.nersc.gov
shifterimg pull registry.nersc.gov/das/delta:4.0
```

## Using at NERSC

There are two main ways we can run Delta in Shifter-- the first is with Delta
installed outside the container, and the second is with Delta install inside
the container.

The image we use is called `registry.nersc.gov/das/delta:4.0`.

This image was built using the `delta-outside` Dockerfile and will use the
installation of Delta outside the container.

## Installing Delta

First, clone this repo on `$SCRATCH`. Then, you'll need to compile the Delta
Cython kernels. The key thing here is that the Cython kernels must be compiled
with the same Delta software installation we will use in the container, so
it must be done inside Shifter.

```
cd $SCRATCH/delta/delta/analysis
shifter --image=registry.nersc.gov/das/delta:4.0 python3 setup.py build_ext --inplace
```

Now we are ready to run Delta.

### Haswell

```
salloc -N 4 -C haswell -q interactive --image=registry.nersc.gov/das/delta:4.0
```

cd to wherever you have installed delta. For me it's

```
cd $SCRATCH/delta/delta
```

And then launch delta:

```
OMP_NUM_THREADS=16 srun -n 16 -c 16 --cpu-bind=cores shifter python3 -m mpi4py.futures processor.py --config configs/hackathon_test.json --transport transport_tx  --num_ranks_preprocess=4 --num_ranks_analysis=12 --num_queue_threads=4 --run_id=test_22289_GT
```

### corigpu

On corigpu, request the image during your interactive job:

```
module load cgpu
salloc -N 1 -C gpu -G 1 -t 60 -c 16 --image=registry.nersc.gov/das/delta:4.0
```

```
OMP_NUM_THREADS=16 srun -n 16 -c 1 --cpu-bind=cores shifter --module=gpu python3 -m mpi4py.futures processor.py --config configs/hackathon_test.json --transport transport_tx  --num_ranks_preprocess=2 --num_ranks_analysis=2 --num_queue_threads=2 --run_id=test_22289_GT
```

Now that I think about it, I'm a little surprised this works at all since it's an MPICH stack. I guess
it still runs on corigpu, but slowly. (Note that we disable the Cray MPICH Shifter module here.)
We may need an OpenMPI based stack if we really need to do serious
corigpu testing.

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
