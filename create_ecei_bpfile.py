#!/usr/bin/env python

"""
Loads the ECEI data from the HDF5 file and converts it into a bp file.


"""

import numpy as np
import h5py
import adios2
from os.path import join


datadir = "/global/cscratch1/sd/rkube/KSTAR/kstar_streaming/018431"

# We have 3 HDF5 file with 192 channels each.
# Each channel has 5,000,000 values.
# Let's divide them so that 10,000 data elements form a time chunk
num_per_chunk = 10000


h5fname_g = join(datadir, "ECEI.018431.GFS.h5")
h5fname_h = join(datadir, "ECEI.018431.HFS.h5")
h5fname_l = join(datadir, "ECEI.018431.LFS.h5")

all_channel_lists = []
for h5_fname in [h5fname_g, h5fname_h, h5fname_l]:
    channel_list = []
    with h5py.File(h5_fname) as h5file:
        for i in h5file["/ECEI"].items():
            channel_list.append(i[0])

            if(len(channel_list) == 1):
                res = h5file["/ECEI/{0:s}/Voltage".format(channel_list[-1])].value
                num_data = res.shape[0]

                num_timesteps = num_data // num_per_chunk
        
        h5file.close()
        all_channel_lists.append(channel_list)


# Continue by creating an adios2 file
adios = adios2.ADIOS(adios2.DebugON)
bpIO = adios.DeclareIO("KSTAR_018431")
bpIO.SetEngine("bp4")

bpWriter = bpIO.Open("KSTAR.bp", adios2.Mode.Write)

# Add all the channels as variables to the bp file
dummy_var = np.zeros(num_per_chunk, dtype=np.float64)
dummy_shape = [num_per_chunk]
dummy_start = [0]
dummy_count = [num_per_chunk]

bp_var_list = []
for ch_list in all_channel_lists:
    for ch_name in ch_list[:3]:
        var = bpIO.DefineVariable(ch_name, dummy_var, dummy_shape, dummy_start, dummy_count, adios2.ConstantDims)
        bp_var_list.append(var)

for tstep in range(num_timesteps):
    t0 = tstep * num_per_chunk
    t1 = (tstep + 1) * num_per_chunk

    print("Time step {0:d}: {1:d} - {2:d}".format(tstep, t0, t1))

    bpWriter.BeginStep()





#bpWriter.close()



