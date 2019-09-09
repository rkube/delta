# -*- coding: UTF-8 -*-

#from mpi4py import MPI
import h5py
from mpi4py import MPI
import numpy as np
#import pickle
import time
import adios2




# """
# Generates batches of ECEI data.
# """

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

shotnr = 18431
data_path = "/global/cscratch1/sd/rkube/KSTAR/kstar_streaming/"
df_fname = "{0:s}/{1:06d}/ECEI.{1:06d}.HFS.h5".format(data_path, shotnr, shotnr)


# # Specify the channel we are streaming
channel = 2202

# # Hard-code the total number of data points
data_pts = int(5e6)
# # Hard-code number of data points per data packet
data_per_batch = int(1e2)
# # Calculate the number of required data batches we send over the channel
num_batches = data_pts // data_per_batch

# # Data to be transported:
data_arr = np.zeros(data_pts, dtype=np.float)



transport_params = {"IPAddress": "127.0.0.1",
                    "Port": "12306"}
#                     "WorkflowMode": "subscribe"}

# # Initialize ADIOS2
adios = adios2.ADIOS(comm)
dataman_IO = adios.DeclareIO("ECEI_H{0:4d}".format(channel))
dataman_IO.SetEngine("DataMan")
dataman_IO.SetParameters(transport_params)
dataman_IO.AddTransport("WAN", transport_params)

# # Open stream
# # Here we have to translate 
# # This is from line 79/80 in helloDataManSubscribeWriter.cpp
# # 79     adios2::Engine dataManWriter =
# # 80         dataManIO.Open("stream", adios2::Mode::Write)
# # adios2::Mode::Write is defined in common/ADIOSTypes.h, I think it is 1.
dataman_writer = dataman_IO.Open("stream", adios2.Mode.Write)

# # Define variable
io_array = dataman_IO.DefineVariable("floats", data_arr, [data_per_batch], [0], [data_per_batch], adios2.ConstantDims)

for i in range(10):
    data_arr = data_arr + 1.0
    dataman_writer.BeginStep()
    dataman_writer.Put(io_array, data_arr, adios2.Mode.Sync)
    dataman_writer.EndStep()
    time.sleep(0.1)

    print(data_arr)



