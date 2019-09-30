#-*- coding: UTF-8 -*-

from mpi4py import MPI
import numpy as np 
import adios2
import json
import argparse 

from readers import reader_dataman, reader_bpfile

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

parser = argparse.ArgumentParser(description="Send KSTAR data using ADIOS2")
parser.add_argument('--config', type=str, help='Lists the configuration file', default='config.json')
args = parser.parse_args()

with open(args.config, "r") as df:
    cfg = json.load(df)
    df.close()

# "Enforce" 1:1 mapping of reader processes on analysis tasks
assert(len(cfg["channels"]) == size)
assert(len(cfg["analysis"]) == size)

datapath = cfg["datapath"]
shotnr = cfg["shotnr"]
channel_list = cfg["channels"]
analysis = cfg["analysis"][rank]





reader = reader_bpfile(shotnr, channel_list[rank])
reader.Open()

print("Starting main loop")

while(True):
    stepStatus = reader.BeginStep()
    print(stepStatus)
    if stepStatus == adios2.StepStatus.OK:
        #var = dataman_IO.InquireVariable("floats")
        #shape = var.Shape()
        #io_array = np.zeros(np.prod(shape), dtype=np.float)
        #reader.Get(var, io_array, adios2.Mode.Sync)
        io_array = reader.get_data("floats")
        #currentStep = reader.CurrentStep()
        reader.EndStep()
        print("Step", reader.CurrentStep(), ", io_array = ", io_array)
    else:
        print("End of stream")
        break


    # Perform the analysis


    # Store result in database

#datamanReader.Close()


# End of file processor_adios2.