#-*- coding: UTF-8 -*-
# Example command: mpirun -n 8 python -u -m mpi4py.futures receiver_mpi.py --config config.json
import numpy as np 
import adios2
import json
import argparse

from analysis.spectral import power_spectrum

import concurrent.futures
import time
import os
import queue
import threading

from mpi4py import MPI
from mpi4py.futures import MPICommExecutor
import sys
from fluctana import *

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,%(msecs)d %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

parser = argparse.ArgumentParser(description="Receive KSTAR data using ADIOS2")
parser.add_argument('--config', type=str, help='Lists the configuration file', default='config.json')
parser.add_argument('--nompi', help='Use with nompi', action='store_true')
parser.add_argument('--debug', help='Use input file to debug', action='store_true')
## A trick to handle: python -u -m mpi4py.futures ...
idx = len(sys.argv) - sys.argv[::-1].index(__file__)
args = parser.parse_args(sys.argv[idx:])

if not args.nompi:
    from processors.readers import reader_dataman, reader_bpfile, reader_sst, reader_gen
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
else:
    from processors.readers_nompi import reader_dataman, reader_bpfile, reader_sst, reader_gen
    comm = None
    rank = 0
    size = 1

with open(args.config, "r") as df:
    cfg = json.load(df)
    df.close()

datapath = cfg["datapath"]
shot = cfg["shot"]
my_analysis = cfg["analysis"][0]
gen_id = 2203 #TODO: Not clear if this was 

#TODO: Remove for non-debug
if args.debug:
    class read_stream(object):
        def __init__(self,shot,nchunk,data_path='./'):
            self.dobj = KstarEcei(shot=shot,data_path=data_path,clist=['ECEI_L0101-2408'],verbose=False)
            self.time = self.dobj.time_base_full()
            tstarts = self.time[::nchunk]
            tstops = self.time[nchunk-1::nchunk]
            if tstarts.size>tstops.size: tstarts = tstarts[:-1]
            self.timeiter = iter(zip(tstarts,tstops))
            self.current_step = 0

        def get_data(self,type_str):
            trange = next(self.timeiter)
            _,data = self.dobj.get_data(trange=trange,norm=1,verbose=0)
            self.current_step += 1
            return trange,data

        def BeginStep(self):
            return True

        def CurrentStep(self):
            return self.current_step

        def EndStep(self):
            pass

    #TODO: Placeholder for data saving
    def save_spec(A):
        pass

    shot = 18431; nchunk=10000
    reader = read_stream(shot=shot,nchunk=nchunk,data_path=datapath)
    #merge into cfg dict
    cfg.update({'shot':shot,'nfft':1000,'window':'hann','overlap':0.0,'detrend':1, 
            'TriggerTime':reader.dobj.tt,'SampleRate':[reader.dobj.fs/1e3], 
            'TFcurrent':reader.dobj.itf/1e3,'Mode':reader.dobj.mode, 
            'LoFreq':reader.dobj.lo,'LensFocus':reader.dobj.sf,'LensZoom':reader.dobj.sz})

#HARDCODED fluctana, does all channels
#number of vertical and radial channels
NV = 24
NR = 8
A = FluctAna()
#TODO: Modify so it can take in a cfg set
dobjAll = KstarEcei(shot=shot,cfg=cfg,clist=['ECEI_L0101-2408'])
A.Dlist.append(dobjAll)  

print("Finished setup")

# Function for workers to perform, which is an analysis.
# Workers (non-master MPI workers) will process each chunk of data.
# mpi4py will be responsible for data distribution.
def perform_analysis(channel_data, step, trange, done_subset, dtwo_subset):
    """ 
    Perform analysis
    """ 
    logging.info(f"\tWorker: do analysis: tstep = {step}, rank = {rank}")
    t0 = time.time()
    if(my_analysis["name"] == "power_spectrum"):
        analysis_result = power_spectrum(channel_data, **my_analysis["config"])
    if(my_analysis["name"] == "all"):
        A.Dlist[0].data = channel_data
        A.Dlist[0].time,_,_,_,_ = A.Dlist[0].time_base(trange)
        #this could be done on rank==0 as Ralph imagined
        A.fftbins(nfft=cfg['nfft'],window=cfg['window'],
              overlap=cfg['overlap'],detrend=cfg['detrend'],full=1)
                #1 cwt
        #TODO: Decide on cwt, need to remove autoplot
        #A.cwt()
        #save_spec(A)
        #2 cross_power
        A.cross_power(done=0,dtwo=0, done_subset=done_subset, dtwo_subset=dtwo_subset)
        save_spec(A) #TODO: determine how to save in aggregate
        #3 coherence
        A.coherence(done=0,dtwo=0, done_subset=done_subset, dtwo_subset=dtwo_subset)
        save_spec(A)
        #4 cross-phase
        A.cross_phase(done=0,dtwo=0, done_subset=done_subset, dtwo_subset=dtwo_subset)
        save_spec(A)
        #5 correlation
        A.correlation(done=0,dtwo=0, done_subset=done_subset, dtwo_subset=dtwo_subset)
        save_spec(A)
        #6 corr_coef
        A.corr_coef(done=0,dtwo=0, done_subset=done_subset, dtwo_subset=dtwo_subset)
        save_spec(A)
        #7 xspec
        #TODO xspec has rnum=cnum, so have to do one at a time, decide best way
        #A.xspec(done=0,dtwo=0, done_subset=done_subset, dtwo_subset=dtwo_subset, plot=False)
        #save_spec(A)
        #8 skw
        #TODO: not same ref/cmp channel setup, check
        #A.skw(done=0,dtwo=0, done_subset=done_subset, dtwo_subset=dtwo_subset, plot=False)
        #save_spec(A)
        #9 bicoherence
        A.bicoherence(done=0,dtwo=0, done_subset=done_subset, dtwo_subset=dtwo_subset, plot=False)
        save_spec(A)
    t1 = time.time()

    # Store result in database
    # backend.store(my_analysis, analysis_result)
    time.sleep(10)
    logging.info(f"\tWorker: done with analysis: tstep = {step}, rank = {rank}, time= {t1-t0}")

# Function for a helper thead (dispatcher).
# The dispatcher will dispatch data in the queue (dq) and 
# distribute to other workers (non-master MPI workers) with mpi4py's MPICommExecutor.
def dispatch():
    while True:
        channel_data, step, trange = dq.get()
        logging.info(f"\tDispatcher: read data: tstep = {step}, rank = {rank}")
        if channel_data is None:
            break
        for ic in range(NV*NR):
            done_subset = [ic]
            dtwo_subset = range(done_subset[0],NV*NR)
            future = executor.submit(perform_analysis, channel_data, step, trange, done_subset, dtwo_subset)
        dq.task_done()

# Main
if __name__ == "__main__":
    with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
        if executor is not None:
            # Only master will execute the following block
            # Use of "__main__" is critical

            # The master thread will keep reading data, while 
            # a helper thread (dispatcher) will dispatch jobs in the queue (dq) asynchronously 
            # and distribute jobs to other workers.
            # The main idea is not to slow down the master.
            dq = queue.Queue()
            dispatcher = threading.Thread(target=dispatch)
            dispatcher.start()

            # Only the master thread will open a data stream.
            # General reader: engine type and params can be changed with the config file
            if not args.debug:
                reader = reader_gen(shot, gen_id, cfg["engine"], cfg["params"])
                reader.Open()

            # Main loop is here
            # Reading data (from KSTAR) and save in the queue (dq) as soon as possible.
            # Dispatcher (a helper thread) will asynchronously fetch data in the queue and distribute to other workers.
            print("Start data reading loop")
            tstart = time.time()
            while(True):
                stepStatus = reader.BeginStep()
                if stepStatus == True:#adios2.StepStatus.OK:
                    trange,channel_data = reader.get_data("floats")
                    currentStep = reader.CurrentStep()
                    reader.EndStep()
                    #print("rank {0:d}: Step".format(rank), reader.CurrentStep(), ", io_array = ", io_array)
                else:
                    logging.info(f"Receiver: end of stream, rank = {rank}")
                    break

                # Recover channel data 
                #TODO: Does the generator.py have to send data over like this?
                # channel_data = channel_data.reshape((num_channels, channel_data.size // num_channels))
                logging.info(f"Receiver: received data step={currentStep}, rank = {rank}")

                # Save data in a queue then go back to work
                # Dispatcher (a helper thread) will fetch asynchronously.
                dq.put((channel_data, currentStep, trange))
                time.sleep(1)
            print("All data read and dispatched, time elapsed: %f" % (time.time()-tstart))
            
            ## Clean up
            dq.join()
            dq.put((None, -1))
            dispatcher.join()
            logging.info(f"Receiver: done")

# End of file processor_adios2.
