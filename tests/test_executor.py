#!/usr/bin/env python
"""
This is a code to test various future execuators; ProcessPoolExecutor, ThreadPoolExecutor, MPICommExecutor, MPIPoolExecutor

Summary
=======

* MPIPoolExecutor 
    - Workers will be created by using MPI Spawn (which is not supported on Cori)
    - After MPI Spawn, workers will have own MPI_COMM_WORLD (which is different from the master's world comm)
    - Workers will execute every things except the "__main__" block after spawning
    - Using "-m mpi4py.futures" in the command will not use MPI Spawn
    - Using "-m mpi4py.futures" in the command will override max_workers argument in MPIPoolExecutor() call
    - If MPI Spawn is enabled, it can be used to construct multiple master-worker sets. 
        e.g.) mpirun -n 2 python -c "MPIPoolExecutor(max_workers=3)" 
            will create a total 6 processes: 2 masters and each of them has 3 workers.
            Note: Using "-m mpi4py.futures" will create only 2 processes in total

* MPICommExecutor 
    - Workers will be created by using MPI comm split
    - MPI collective calls can be used outside of the "__main__" block as well as outside of the MPICommExecutor context.

* Using "-m mpi4py.futures" in the command line
    - mpi4py will automatically split n MPI processes into two groups; a master (rank 0) and n-1 workers.
    - n-1 workers will be blocked until the master execute MPICommExecutor/MPIPoolExecutor calls (submit, map, etc),
    which cause any MPI collective calls (barrier/send/receive/etc) to be hanged.
    - n-1 workers will not execute the "__main__" block

Command
=======
```
$ python test_executor.py -h                                                                      
usage: test_executor.py [-h] [--nworkers NWORKERS]
                        [--ncorespernode NCORESPERNODE] [--nsteps NSTEPS]
                        [--chunksize CHUNKSIZE] [--nanalysis NANALYSIS]
                        [--checkclock] [--setaffinity]
                        [--processpool | --threadpool | --mpicomm | --mpipool]

optional arguments:
  -h, --help            show this help message and exit
  --nworkers NWORKERS   Number of workers
  --ncorespernode NCORESPERNODE
                        Number of cores per node
  --nsteps NSTEPS       Number of steps
  --chunksize CHUNKSIZE
                        Number of steps in data chunk
  --nanalysis NANALYSIS
                        Number of nanalysis
  --checkclock          Check clock diff
  --setaffinity         Set affinity
  --processpool         use ProcessPoolExecutor
  --threadpool          use ThreadPoolExecutor
  --mpicomm             use MPICommExecutor
  --mpipool             use MPIPoolExecutor
```

Examples (Cori)
========
* Run MPIPoolExecutor wtth 8 workers (will fail with MPID_Comm_spawn_multiple not implemented)
    srun -n 1 python test_executor.py --mpipool --nworkers=8
  Intead, run:
    srun -n 9 python -m mpi4py.futures test_executor.py --mpipool

* Run MPICommExecutor wtth 1 master-8 workers. The following two commands will same:
    srun -n 9 python test_executor.py --mpicomm
    srun -n 9 python -m mpi4py.futures test_executor.py --mpicomm

  However, the following will hang since `checkclock` option will call MPI send/receive
    srun -n 9 python -m mpi4py.futures test_executor.py --mpicomm --checkclock
  The following will work:
    srun -n 9 python test_executor.py --mpicomm --checkclock

* ProcessPoolExecutor with and without affinity (recommend to check with htop)
    srun -n 1 python test_executor.py --processpool --nworkers=8 --setaffinity
    srun -n 1 python test_executor.py --processpool --nworkers=8

* ThreadPoolExecutor with and without affinity (recommend to check with htop)
    srun -n 1 python test_executor.py --threadpool --nworkers=8 --setaffinity
    srun -n 1 python test_executor.py --threadpool --nworkers=8

"""

import numpy as np 
import adios2

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

from mpi4py import MPI
from mpi4py.futures import MPICommExecutor, MPIPoolExecutor

import time
import os
import socket
import queue
import threading

import logging
import sys
import argparse

def check_clockdiff():
    def _SKaMPI_pingpong(p1, p2, n_pingpongs=100) -> float:

        td_min = -float("inf")
        td_max = float("inf")

        for i in range(n_pingpongs):
            if (rank == p1):
                s_last = time.time()
                comm.send(s_last, dest=p2)
                t_last = comm.recv(source=p2)
                s_now = time.time()

                td_min = max(td_min, t_last - s_now)
                td_max = min(td_max, t_last - s_last)
            elif (rank == p2):
                s_last = comm.recv(source=p1)
                t_last = time.time()
                comm.send(t_last, dest=p1)
                t_now = time.time()

                td_min = max(td_min, s_last - t_now)
                td_max = min(td_max, s_last - t_last)
        
        diff = (td_min + td_max)/2.0
        #print (">> rank, td_min, td_max, diff= %d\t%.09f\t%.09f\t%.09f"%(rank, td_min, td_max, diff))

        return diff

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    diff = [0.0,]*size

    for i in range(1,size):
        comm.barrier()
        if rank == 0:
            diff[i] = _SKaMPI_pingpong(i, 0)
        elif rank == i:
            diff[i] = _SKaMPI_pingpong(i, 0)
    
    return diff[rank]

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

parser = argparse.ArgumentParser()
# parser.add_argument('--config', type=str, help='Lists the configuration file', default='config.json')
parser.add_argument('--nworkers', type=int, help='Number of workers', default=1)
parser.add_argument('--ncorespernode', type=int, help='Number of cores per node', default=64)
parser.add_argument('--nsteps', type=int, help='Number of steps', default=100)
parser.add_argument('--chunksize', type=int, help='Number of steps in data chunk', default=10000)
parser.add_argument('--nanalysis', type=int, help='Number of nanalysis', default=100)
parser.add_argument('--ntasks', type=int, help='Number of tasks (separate executor launches / data chunk)', default=1)
parser.add_argument('--checkclock', help='Check clock diff', action='store_true')
parser.add_argument('--setaffinity', help='Set affinity', action='store_true')
group = parser.add_mutually_exclusive_group()
group.add_argument('--processpool', help='use ProcessPoolExecutor', action='store_const', dest='pool', const='process')
group.add_argument('--threadpool', help='use ThreadPoolExecutor', action='store_const', dest='pool', const='thread')
group.add_argument('--mpicomm', help='use MPICommExecutor', action='store_const', dest='pool', const='mpicomm')
group.add_argument('--mpipool', help='use MPIPoolExecutor', action='store_const', dest='pool', const='mpipool')
parser.set_defaults(pool='process')

## A trick to handle: python -u -m mpi4py.futures ...
idx = len(sys.argv) - sys.argv[::-1].index(__file__)
args = parser.parse_args(sys.argv[idx:])

hostname = socket.gethostname()

## dict for pid-workerid. We set with an initial value (rank) but it can be replaced by diffrent numbers laters.
pidmap = {}
pidmap[os.getpid()] = rank
print ('Has parent?', comm.Get_parent() != MPI.COMM_NULL, rank, size, os.getpid())

## arg is self. just pass arg2 to time.localtime
def myrelativetime(arg=None, arg2=None):
    global time_drift
    return time.localtime(arg2+time_drift)

logging.Formatter.converter = myrelativetime
logging.basicConfig(
    level=logging.INFO,
    format="%%(asctime)s,%%(msecs)d %%(levelname)s (rank %d): %%(message)s"%(rank),
    datefmt="%H:%M:%S",
)

## This is to adjust clock skewness between multiple nodes.
## We measure clock differences and use with logging
## This uses MPI send/receive, which conflicts with MPIPoolExecutor
time_drift = 0.0
if args.checkclock:
    time_drift = check_clockdiff()
    logging.info(f"Time drift: {time_drift}")

# Function for workers to perform, which is an analysis.
# Workers (non-master MPI workers) will process each chunk of data.
# mpi4py will be responsible for data distribution.
def perform_analysis(tstep, channel_data):
    """ 
    Perform analysis
    """ 
    logging.info(f"\tWorker: perform_analysis start: tstep={tstep} rank={rank} pid={os.getpid()}")
    
    size = 1024
    A, B = np.random.random((size, size)), np.random.random((size, size))

    # Matrix multiplication
    t0 = time.time()
    for i in range(args.nanalysis):
        np.dot(A, B)
    t1 = time.time()
    logging.info(f"\tWorker: perform_analysis done: tstep={tstep} rank={rank} pid={os.getpid()} ID={pidmap[os.getpid()]} hostname={hostname} time elapsed: {t1-t0:.2f}")
    return tstep

# Function for a helper thead (dispatcher).
# The dispatcher will dispatch data in the queue (dq) and 
# distribute to other workers (non-master MPI workers) with mpi4py's MPICommExecutor.
# We assume only a single dispatcher
def dispatch():
    isfirst = True
    while True:
        tstep, channel_data = dq.get()
        logging.info(f"\tDispatcher: read data: tstep={tstep}, rank={rank}")
        if channel_data is None:
            dq.task_done()
            logging.info(f"\tDispatcher: no more data. break. rank={rank}")
            break
        for i in range(args.ntasks): 
            future = executor.submit(perform_analysis, tstep, channel_data)
            fs.put(future)
        dq.task_done()

def foo(n):
    time.sleep(2)
    return n

def hello(counter):
    with counter.get_lock():
        counter.value += 1
        pidmap[os.getpid()] = (args.nworkers+1)*rank + counter.value
    affinity = None
    ## Set affinity when using ProcessPoolExecutor
    if args.pool == 'process':
        if hasattr(os, 'sched_getaffinity'):
            if args.setaffinity:
                ## We leave rank-0 core for the main process
                affinity_mask = {pidmap[os.getpid()]%args.ncorespernode}
                os.sched_setaffinity(0, affinity_mask)
            affinity = os.sched_getaffinity(0)
    logging.info(f"\tWorker: init. rank={rank} pid={os.getpid()} hostname={hostname} ID={pidmap[os.getpid()]} affinity={affinity}")
    # time.sleep(random.randint(1, 5))
    return 0

def hello_mpi(counter):
    counter.value = rank
    pidmap[os.getpid()] = counter.value
    affinity = None
    logging.info(f"\tWorker: init. rank={rank} pid={os.getpid()} hostname={hostname} ID={pidmap[os.getpid()]} affinity={affinity}")

# Main
if __name__ == "__main__":


    ## Generating dummy data
    if rank == 0:
        logging.info("Command: {0}".format(" ".join([x for x in sys.argv])))
        logging.info("All settings used:")
        for k,v in sorted(vars(args).items()):
            logging.info("\t{0}: {1}".format(k,v))

        logging.info(f"Dumping data")
        adios = adios2.ADIOS(MPI.COMM_SELF)
        IO = adios.DeclareIO('write')
        data_array = np.ones((192, args.chunksize))
        v1 = IO.DefineVariable('tstep', np.array(1))
        v2 = IO.DefineVariable('data', data_array,
                                    data_array.shape, # shape
                                    list(np.zeros_like(data_array.shape, dtype=int)),  # start 
                                    data_array.shape, # count
                                    adios2.ConstantDims)

        writer = IO.Open('test.bp', adios2.Mode.Write)

        for i in range(args.nsteps):
            writer.BeginStep()
            writer.Put(v1, np.array(i))
            writer.Put(v2, data_array)
            writer.EndStep()

        writer.Close()

    ## To ensure to have unique ID for multiple processes or threads
    counter = mp.Value('i', 0)
    if args.pool == 'process':
        logging.info(f"Using: ProcessPoolExecutor")
        pool = ProcessPoolExecutor(max_workers=args.nworkers, initializer=hello, initargs=(counter,))

    if args.pool == 'thread':
        logging.info(f"Using: ThreadPoolExecutor")
        pool = ThreadPoolExecutor(max_workers=args.nworkers, initializer=hello, initargs=(counter,))

    if args.pool == 'mpicomm':
        logging.info(f"Using: MPICommExecutor")
        pool = MPICommExecutor(comm)
        hello_mpi(counter)

    if args.pool == 'mpipool':
        logging.info(f"Using: MPIPoolExecutor")
        ## Note: max_workers will be overrite when -m mpi4py.future given in the command line
        ## Note: max_workers includes the master (rank 0) process too (others not.)
        pool = MPIPoolExecutor(max_workers=args.nworkers)
        hello_mpi(counter)

    with pool as executor:
        if executor is not None:
            # Only master will execute the following block
            # Use of "__main__" is critical
            logging.info(f"\tMaster: init. rank={rank} pid={os.getpid()} hostname={hostname} ID={pidmap[os.getpid()]}")

            # The master thread will keep reading data, while 
            # a helper thread (dispatcher) will dispatch jobs in the queue (dq) asynchronously 
            # and distribute jobs to other workers.
            # The main idea is not to slow down the master.
            dq = queue.Queue()
            
            dispatcher = threading.Thread(target=dispatch)
            dispatcher.start()

            # We use another queue to track output (filled with future objects)
            fs = queue.Queue()


            ## Check if all workers are successfully created.
            if args.pool in ('process','thread'):
                # Warming-up (just to make sure workers are created before running main analysis)
                for _ in executor.map(foo, range(2*args.nworkers)):
                    pass
                time.sleep(3)

                while True:
                    with counter.get_lock():
                        logging.info(f'nworkers so far {counter.value}')
                        if counter.value == args.nworkers:
                            break
                        else:
                            time.sleep(1)

            ## Read open
            adios = adios2.ADIOS(MPI.COMM_SELF)
            IO = adios.DeclareIO('read')
            reader = IO.Open('test.bp', adios2.Mode.Read)

            # Main loop is here
            # Reading data (from KSTAR) and save in the queue (dq) as soon as possible.
            # Dispatcher (a helper thread) will asynchronously fetch data in the queue and distribute to other workers.
            logging.info(f"Start data reading loop: pid={os.getpid()}")
            t0 = time.time()
            isfirst = True
            n = 0
            while(True):
                stepStatus = reader.BeginStep()
                if stepStatus == adios2.StepStatus.OK:
                    ## Set a timer when we receive the first chunk
                    if isfirst:
                        t1 = time.time()
                        isfirst = False
                    
                    var = IO.InquireVariable('tstep')
                    currentStep = np.zeros(var.Shape(), dtype=np.int)
                    reader.Get(var, currentStep, adios2.Mode.Sync)
                    var = IO.InquireVariable('data')
                    channel_data = np.zeros(var.Shape(), dtype=np.float64)
                    reader.Get(var, channel_data, adios2.Mode.Sync)
                    reader.EndStep()
                else:
                    logging.info(f"Receiver: end of stream, status={stepStatus}")
                    break
                logging.info(f"Receiver: received data tstep={currentStep}")

                # Save data in a queue then go back to work
                # Dispatcher (a helper thread) will fetch asynchronously.
                dq.put((currentStep, channel_data))

            t2 = time.time()
            logging.info(f"All data read and dispatched, time elapsed: {t2-t1:.2f}")
            
            ## Clean up
            dq.join()
            dq.put((-1, None))
            dispatcher.join()
            fs.put(None)
            logging.info(f"All done.")

            logging.info(f"Futures: len={fs.qsize()}")
            for i in range(fs.qsize()):
            #for future in iter(fs.get, None):
                future = fs.get()
                if future is not None:
                    logging.info(f"future done? {future.result()}")
                else:
                    logging.info(f"no more")
                    break

    ## All done
    t3 = time.time()
    if (args.pool.startswith('mpi') and (rank==0)) or (args.pool == 'process') or (args.pool == 'thread'): 
        logging.info(f"Receiver: done, time elapsed: {t3-t1:.2f}")
        logging.info(f"")
        logging.info(f"Summary:")
        logging.info(f"Data waiting time: {t1-t0:.2f}")
        logging.info(f"Data loading and queuing time: {t2-t1:.2f}")
        logging.info(f"Overall time: {t3-t1:.2f}")

# End of file processor_adios2.
