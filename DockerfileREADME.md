# README for Delta Dockerfile

The Dockerfile is the blueprint for the delta image. It is built on top of an
NVIDIA cuda image to provide GPU support. It largely follows the instructions
here: https://delta-fusion.readthedocs.io/en/latest/notes/installing.html

## Using at NERSC

The image is called `registry.nersc.gov/das/delta:3.0`. 

As a backup, there is also an image that has delta installed inside: `registry.nersc.gov/das/delta-inside:1.0`

On corigpu, request the image during your interactive job:

```
module load cgpu
salloc -N 1 -C gpu -G 1 -t 120 -A m499 --image=registry.nersc.gov/das/delta:3.0
```

On Cori, `module unload cgpu` and adjust your salloc/sbatch accordingly.

To open a shell inside the container and run tests/scripts inside:

```
shifter --volume="/global/cscratch1/sd/stephey/delta:/opt/delta" /bin/bash
```

The `--volume` command will bind-mount your delta installation into
`/opt/delta` inside the image. Note that the bind location `/opt/delta` must
exist inside the image.

To use shifter to run your tests/scripts:

shifter --volume="/global/cscratch1/sd/stephey/delta:/opt/delta" ls /opt/delta/tests
```
