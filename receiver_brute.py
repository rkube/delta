#-*- coding: UTF-8 -*-
# Example command: srun -n 8 python -u -m mpi4py.futures receiver_brute_mpi.py --config config.receiver.json
# this code requires the modified fluctana code, https://github.com/rmchurch/fluctana.git

import numpy as np 
import adios2
import json
import argparse

import concurrent.futures
import time
import os
import socket
import queue
import threading

import sys
from fluctana import *

from itertools import combinations 

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
parser.add_argument('--middleman', help='Run as a middleman', action='store_true')
parser.add_argument('--nworkers', type=int, help='Number of workers to middle', default=1)
parser.add_argument('--workwithmiddleman', help='Process with middleman', action='store_true')
parser.add_argument('--ngroups', type=int, help='Number of subgroups', default=1)
parser.add_argument('--subjob', help='subjob', action='store_true')
parser.add_argument('--onlyn', type=int, help='process only n')
parser.add_argument('--blocksize', type=int, help='blocksize', default=24)
## A trick to handle: python -u -m mpi4py.futures ...
idx = len(sys.argv) - sys.argv[::-1].index(__file__)
args = parser.parse_args(sys.argv[idx:])

if not args.nompi:
    from processors.readers import reader_gen
    from generators.writers import writer_gen
    from mpi4py import MPI
    from mpi4py.futures import MPICommExecutor
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
else:
    from generators.writers_nompi import writer_gen
    from concurrent.futures import ProcessPoolExecutor as MPICommExecutor
    comm = None
    rank = 0
    size = 1
hostname = socket.gethostname()

with open(args.config, "r") as df:
    cfg = json.load(df)
    df.close()


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

        def get_all_data(self):
            _,data = self.dobj.get_data(trange=[self.time[0],self.time[-1]],norm=1,verbose=0)
            n = int(np.ceil(data.shape[-1]/nchunk))
            self.dataSplit = np.array_split(data,n,axis=-1)

        def get_data(self,type_str):
            if 'trange' in type_str:
                trange = next(self.timeiter)
            else:
                #_,data = self.dobj.get_data(trange=trange,norm=1,verbose=0)
                data = self.dataSplit[self.current_step]
                self.current_step += 1
                return data

        def BeginStep(self):
            if (self.current_step<len(self.dataSplit)):
                return adios2.stepStatus.OK

        def CurrentStep(self):
            return self.current_step

        def EndStep(self):
            pass


    shot = 18431; nchunk=10000
    reader = read_stream(shot=shot,nchunk=nchunk,data_path=cfg["datapath"])
    #merge into cfg dict
    cfg.update({'shot':shot,'nfft':1000,'window':'hann','overlap':0.0,'detrend':1, 
            'TriggerTime':reader.dobj.tt,'SampleRate':[reader.dobj.fs/1e3], 
            'TFcurrent':reader.dobj.itf/1e3,'Mode':reader.dobj.mode, 
            'LoFreq':reader.dobj.lo,'LensFocus':reader.dobj.sf,'LensZoom':reader.dobj.sz})

def writer_init(shotnr, gen_id, worker_id, data_arr):
    logging.info(f"\tMiddleman: opening a channel for writer #{worker_id}")
    writer = writer_gen(shotnr, gen_id, cfg["middleman_engine"], cfg["middleman_params"])

    writer.DefineVariable("tstep",np.array(0))
    writer.DefineVariable("floats",data_arr)
    writer.DefineVariable("trange",np.array([0.0,0.0]))
    writer.DefineAttributes("cfg",cfg)
    writer.Open(worker_id=worker_id)
    return writer

def save_spec(results,tstep):
    #TODO: Determine how to use adios2 efficiently instead (and how to read in like normal, e.g. without steps?)
    #np.savez(resultspath+'delta.'+str(tstep).zfill(4)+'.npz',**results)
    with adios2.open(cfg["resultspath"]+'delta.'+str(tstep).zfill(4)+'.bp','w') as fw:
        for key in results.keys():
            fw.write(key,results[key],results[key].shape,[0]*len(results[key].shape),results[key].shape)


A = FluctAna(verbose=False)
#TODO: Modify so it can take in a cfg set
#dobjAll = KstarEcei(shot=shot,cfg=cfg,clist=['ECEI_L0101-2408'],verbose=False)
#A.Dlist.append(dobjAll)  

# Function for workers to perform, which is an analysis.
# Workers (non-master MPI workers) will process each chunk of data.
# mpi4py will be responsible for data distribution.
def perform_analysis(channel_data, cfg, tstep, trange):
    """ 
    Perform analysis
    """ 
    logging.info(f"\tWorker: start perform_analysis: tstep = {tstep}, rank = {rank}")
    t0 = time.time()
    if(cfg["analysis"][0]["name"] == "all"):
        results = {} 
        dobjAll = KstarEcei(shot=cfg["shotnr"],cfg=cfg,clist=cfg["channel_range"],verbose=False)
        if len(A.Dlist)==0: 
            A.Dlist.append(dobjAll)
        else:
            A.Dlist[0] = dobjAll
        A.Dlist[0].data = channel_data
        A.Dlist[0].time,_,_,_,_ = A.Dlist[0].time_base(trange)
        #this could be done on rank==0 as Ralph imagined
        A.fftbins(nfft=cfg['fft_params']['nfft'],window=cfg['fft_params']['window'],
          overlap=cfg['fft_params']['overlap'],detrend=cfg['fft_params']['detrend'],full=1,scipy=True)
        results['stft'] = A.Dlist[0].spdata

        Nchannels = channel_data.shape[0] 
        for ic in range(Nchannels):
            #logging.info(f"\tWorker: do analysis: tstep={tstep}, rank={rank}, analysis={ic}, hostname={hostname}")
            chstr = A.Dlist[0].clist[ic]
            done_subset = [ic]
            dtwo_subset = range(done_subset[0],Nchannels)
            #TODO: Decide on cwt, need to remove autoplot
            #1 cwt
            #A.cwt()
            #save_spec(A)
            #2 cross_power
            A.cross_power(done=0,dtwo=0, done_subset=done_subset, dtwo_subset=dtwo_subset)
            results['cross_power/'+chstr] = A.Dlist[0].val
            #3 coherence
            A.coherence(done=0,dtwo=0, done_subset=done_subset, dtwo_subset=dtwo_subset)
            results['coherence/'+chstr] = A.Dlist[0].val
            #4 cross-phase
            A.cross_phase(done=0,dtwo=0, done_subset=done_subset, dtwo_subset=dtwo_subset)
            results['cross_phase/'+chstr] = A.Dlist[0].val
            #5 correlation
            A.correlation(done=0,dtwo=0, done_subset=done_subset, dtwo_subset=dtwo_subset)
            results['correlation/'+chstr] = A.Dlist[0].val
            #6 corr_coef
            A.corr_coef(done=0,dtwo=0, done_subset=done_subset, dtwo_subset=dtwo_subset)
            results['corr_coef/'+chstr] = A.Dlist[0].val
            #7 xspec
            #TODO xspec has rnum=cnum, so have to do one at a time, decide best way
            #A.xspec(done=0,dtwo=0, done_subset=done_subset, dtwo_subset=dtwo_subset, plot=False)
            #save_spec(A)
            #8 skw
            #TODO: not same ref/cmp channel setup, check
            #A.skw(done=0,dtwo=0, done_subset=done_subset, dtwo_subset=dtwo_subset, plot=False)
            #save_spec(A)
            #9 bicoherence
            #A.bicoherence(done=0,dtwo=0, done_subset=done_subset, dtwo_subset=dtwo_subset, plot=False)
            #results['bicoherence/'+chstr] = A.Dlist[0].val
            # Store result in database
            # backend.store(my_analysis, analysis_result)
            #logging.info(f"\tWorker: loop done: tstep={tstep}, rank={rank}, analysis={ic}, hostname={hostname}")
        t1 = time.time()
        #save_spec(results,tstep)
        t2 = time.time()
        logging.info(f"\tWorker: perform_analysis done: tstep={tstep}, rank={rank}, hostname={hostname} time elapsed: {t2-t0:.2f}")

# Function for a helper thead (dispatcher).
# The dispatcher will dispatch data in the queue (dq) and 
# distribute to other workers (non-master MPI workers) with mpi4py's MPICommExecutor.
# We assume only a single dispatcher
def dispatch():
    isfirst = True
    while True:
        channel_data, cfg, tstep, trange = dq.get()
        logging.info(f"\tDispatcher: read data: tstep = {tstep}, rank = {rank}")
        if channel_data is None:
            break
        
        ## Act as a middleman
        ## Middleman receives data from the generator and distribute to n processors
        ## We need to open channel only after receiving at least one step
        if isfirst and args.middleman:
            nworkers = args.nworkers
            writer_list = list()
            for i in range(nworkers):
                shotnr = cfg["shotnr"]
                gen_id = 100000 * rank
                channels = expand_clist(cfg["channel_range"])
                batch_size = cfg['batch_size']
                data_array = np.zeros((len(channels), batch_size), dtype=np.float64)
                w = writer_init(shotnr, gen_id, i, data_array)
                writer_list.append(w)
            isfirst = False

        ## If middleman, we write data. Otherwise, distribute to others for analysis
        if args.middleman:
            writer = writer_list[tstep%args.nworkers]
            with writer.step() as w:
                w.put_data("tstep",np.array(tstep))
                w.put_data("floats",channel_data)
                w.put_data("trange",np.array(trange))
        else:
            future = executor.submit(perform_analysis, channel_data, cfg, tstep, trange)
        dq.task_done()

    ## Once done, we close open files for writing
    if args.middleman:
        nworkers = args.nworkers
        for i in range(nworkers):
            writer = writer_list[i]
            writer.writer.Close()

# Main
if __name__ == "__main__":

    # ## Create a subcomm
    # color = rank*args.ngroups//size
    # key = rank
    # print ('rank, color, key:', rank, color, key)
    # newcomm = comm.Split(color, key)

    # newrank = newcomm.Get_rank()
    # newsize = newcomm.Get_size()
    # print ('rank, newrank, newsize:', rank, newrank, newsize)

    # ## Act as a middleman
    # ## Middleman receives data from the generator and distribute to n processors
    # if args.middleman:
    #     nworkers = args.nworkers
    #     writer_list = list()
    #     for i in range(nworkers):
    #         shotnr = cfg["shotnr"]
    #         gen_id = 100000 * rank
    #         channels = expand_clist(cfg["channel_range"])
    #         batch_size = cfg['batch_size']
    #         data_array = np.zeros((len(channels), batch_size), dtype=np.float64)
    #         w = writer_init(shotnr, gen_id, i, data_array)
    #         writer_list.append(w)

    with MPICommExecutor(comm, root=0) as executor:
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
                if args.workwithmiddleman:
                    reader = reader_gen(cfg["shotnr"], 0, cfg["middleman_engine"], cfg["middleman_params"])
                    reader.Open(worker_id=0)
                else:
                    reader = reader_gen(cfg["shotnr"], 0, cfg["engine"], cfg["params"])
                    reader.Open()
            else:
                reader.get_all_data()

            # Main loop is here
            # Reading data (from KSTAR) and save in the queue (dq) as soon as possible.
            # Dispatcher (a helper thread) will asynchronously fetch data in the queue and distribute to other workers.
            cfg_update = False
            logging.info(f"Start data reading loop")
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
                    #currentStep = reader.CurrentStep()
                    currentStep = reader.get_data("tstep")
                    logging.info(f"Step {currentStep} started")
                    trange = list(reader.get_data("trange"))
                    channel_data = reader.get_data("floats")
                    if not cfg_update:
                        cfg.update(reader.get_attrs("cfg"))
                        cfg_update = True

                        # ## Act as a middleman
                        # ## Middleman receives data from the generator and distribute to n processors
                        # if args.middleman:
                        #     nworkers = args.nworkers
                        #     writer_list = list()
                        #     for i in range(nworkers):
                        #         shotnr = cfg["shotnr"]
                        #         gen_id = 100000 * rank
                        #         channels = expand_clist(cfg["channel_range"])
                        #         batch_size = cfg['batch_size']
                        #         data_array = np.zeros((len(channels), batch_size), dtype=np.float64)
                        #         w = writer_init(shotnr, gen_id, i, data_array)
                        #         writer_list.append(w)
                        
                    reader.EndStep()
                else:
                    logging.info(f"Receiver: end of stream, rank = {rank}")
                    break

                # Recover channel data 
                #TODO: Does the generator.py have to send data over like this?
                # channel_data = channel_data.reshape((num_channels, channel_data.size // num_channels))
                logging.info(f"Receiver: received data tstep={currentStep}, rank = {rank}")

                # Save data in a queue then go back to work
                # Dispatcher (a helper thread) will fetch asynchronously.
                #dq.put((channel_data, cfg, currentStep, trange))
                #perform_analysis(channel_data, cfg, currentStep, trange)
                ## jyc: Testing decomposition
                if args.subjob:
                    blocksize = args.blocksize
                    comb = list()
                    for i,j in combinations(range(channel_data.shape[0]//blocksize), 2):
                        comb.append((i,j))
                    for i in range(channel_data.shape[0]//blocksize):
                        comb.append((i,i))
                    logging.info(f"Decomposition: created {len(comb)} subjobs")
                    for i,j in comb:
                        logging.info(f"Decomposition: received data tstep={currentStep}, rank = {rank}, ({i},{j})")
                        np.r_[channel_data[i:8,:], channel_data[0:8,:]].shape
                        block1 = channel_data[i*blocksize:i*blocksize+blocksize,:]
                        block2 = channel_data[j*blocksize:j*blocksize+blocksize,:]
                        data = np.r_[block1,block2]
                        dq.put((data, cfg, currentStep, trange))
                else:
                    dq.put((channel_data, cfg, currentStep, trange))
                    #perform_analysis(channel_data, cfg, currentStep, trange)

                n = n + 1
                if (args.onlyn is not None) and (n >= args.onlyn):
                    break
            t2 = time.time()
            logging.info(f"All data read and dispatched, time elapsed: {t2-t1:.2f}")
            
            ## Clean up
            dq.join()
            dq.put((None, None, -1, -1))
            dispatcher.join()

    ## All done
    t3 = time.time()
    if (rank==0): 
        logging.info(f"Receiver: done, time elapsed: {t3-t1:.2f}")
        logging.info(f"")
        logging.info(f"Summary:")
        logging.info(f"Data waiting time: {t1-t0:.2f}")
        logging.info(f"Data loading and queuing time: {t2-t1:.2f}")
        logging.info(f"Overall time: {t3-t1:.2f}")

# End of file processor_adios2.
