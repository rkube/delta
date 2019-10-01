#-*- coding: UTF-8 -*-

from mpi4py import MPI
import numpy as np 
import adios2
import json
import argparse


from processors.readers import reader_dataman, reader_bpfile
from analysis.spectral import power_spectrum

from backends.mongodb import mongo_backend


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
assert(len(cfg["channel_lists"]) == size)
assert(len(cfg["analysis"]) == size)

datapath = cfg["datapath"]
shotnr = cfg["shotnr"]
my_analysis = cfg["analysis"][rank]
my_channel_list = cfg["channel_lists"][rank]
gen_id = 100000 * rank + my_channel_list[0]
num_channels = len(my_channel_list)

reader = reader_bpfile(shotnr, gen_id)
reader.Open()


backend = mongo_backend(rank, my_channel_list)

#print("Starting main loop")

while(True):
    stepStatus = reader.BeginStep()
    #print(stepStatus)
    if stepStatus == adios2.StepStatus.OK:
        #var = dataman_IO.InquireVariable("floats")
        #shape = var.Shape()
        #io_array = np.zeros(np.prod(shape), dtype=np.float)
        #reader.Get(var, io_array, adios2.Mode.Sync)
        channel_data = reader.get_data("floats")
        #currentStep = reader.CurrentStep()
        reader.EndStep()
        #print("rank {0:d}: Step".format(rank), reader.CurrentStep(), ", io_array = ", io_array)
    else:
        print("rank {0:d}: End of stream".format(rank))
        break

    # Recover channel data 
    channel_data = channel_data.reshape((num_channels, channel_data.size // num_channels))

    # Perform the analysis
    if(my_analysis["name"] == "power_spectrum"):
        analysis_result = power_spectrum(io_array, **my_analysis["config"])

    # Store result in database
    backend.store(my_analysis, analysis_result)

#datamanReader.Close()


# End of file processor_adios2.