#-*- coding: UTF-8 -*-

"""
This data processor implements the many-to-many model.
It runs as an MPI application and assumes that the sender sends data on a number of 
channels equal to the number of MPI ranks.

The config file specifies a list of analysis tasks. The length of this lists must correspond to the
number of MPI ranks.

Each MPI rank receives data as chunks in time. For each time chunk, the specified analysis task is performed.
The results are stored in a database
"""
from mpi4py import MPI
import numpy as np 
import adios2
import json
import argparse


from processors.readers_many_to_many import reader_many_to_many_dataman, reader_many_to_many_bpfile
from analysis.spectral import power_spectrum

from backends.mongodb import mongo_backend


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

parser = argparse.ArgumentParser(description="Send KSTAR data using ADIOS2")
parser.add_argument('--config', type=str, help='Lists the configuration file', default='config_many_to_many.json')
args = parser.parse_args()

with open(args.config, "r") as df:
    cfg = json.load(df)
    df.close()

# "Enforce" 1:1 mapping of reader processes on analysis tasks
assert(len(cfg["channel_ranges"]) == size)
assert(len(cfg["analysis"]) == size)

datapath = cfg["datapath"]
shotnr = cfg["shotnr"]
my_analysis = cfg["analysis"][rank]
my_channel_range = cfg["channel_ranges"][rank]
gen_id = 100000 * rank + my_channel_range[0]
num_channels = len(my_channel_range)

reader = reader_many_to_many_bpfile(shotnr, gen_id)
reader.Open()

backend = mongo_backend(rank, my_channel_range)

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