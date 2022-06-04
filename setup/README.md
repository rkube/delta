# Helper scripts for building and setting up Delta on the NERSC DTNs

Since the NERSC DTNs don't currently support Shifter, we'll need to build
our Delta stack manually.

`build-delta-dtn.sh` builds Delta and all of its dependencies. It assumes
you already have a local clone of Delta and will use the requirements file
it finds there.

`setup-delta-dtn.sh` assumes you have built Delta using the
`build-delta-dtn.sh` script. You can 

`source setup-delta-dtn.sh` to activate the custom Delta conda environment
and point to the right Adios2 Python libraries.

## Notes

We're using normal MPICH here (not optimized Cray MPICH) since we don't have
a high performance network on the DTNs. This hopefully lets us maintain compatbility
with the rest of the Delta stack.

We are trying to keep this build in line with the current iteration of the Delta
dockerfile [here](https://github.com/rkube/delta/blob/master/container/delta-outside/Dockerfile).
Of course this is all done by hand so updates to the Dockerfile will eventually
need to be made here, too.
