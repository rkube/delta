# README for Delta Dockerfile

The Dockerfile is the blueprint for the delta image. It is built on top of an NVIDIA cuda image to provide GPU support. It largely follows the instructions here: https://delta-fusion.readthedocs.io/en/latest/notes/installing.html

## Pulling

If you want to just pull and use the pre-built container

```
docker pull stephey/delta:1.0
```

You can find more information here: https://hub.docker.com/repository/docker/stephey/delta

## Building

To build the container locally, make sure you have Docker installed on your machine. You may also want to sign up for a free dockerhub account.

```
docker build -t <your dockerhub username>/delta:1.0 .
```

here we chose the `1.0` tag but you can call it whatever you like.

Note that building the container requires the `requirements.txt` file present alongside the Dockerfile to be copied into the image.

Building from scratch takes ~30 mins.

## Running

To run the container and look around inside:

```
docker run -it --rm <your dockerhub username>/delta:1.0 /bin/bash
```

## Pushing

To push the container to dockerhub:

```
docker login
#enter your dockerhub credentials
docker push <your dockerhub username>/delta:1.0
```

Depending on your upload speed, this can be very slow (~hours). The next push will be faster since it can build on existing images.

## Using

On NERSC, we require that Docker containers are run using Shifter for security reasons (i.e. no root privileges.) You can find more info here: https://docs.nersc.gov/development/shifter/how-to-use/

Notes: we have had some trouble with Python + CUDA drivers in shifter containers. We may also see this in the delta container. We're working on a fix for this.

On other systems and clusters, you may be able to run using Docker directly or nvidia-docker.


