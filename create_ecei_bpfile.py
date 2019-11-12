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

chlist_g = []
chlist_h = []
chlist_l = []



# First we gather all the different channel names
for h5_fname, chlist in zip([h5fname_g, h5fname_h, h5fname_l], 
                            [chlist_g, chlist_h, chlist_l]):
    with h5py.File(h5_fname) as h5file:
        for i in h5file["/ECEI"].items():
            chlist.append(i[0])
            if(len(chlist) == 1):
                num_data = h5file["/ECEI/{0:s}/Voltage".format(chlist[-1])].shape[0]
                num_timesteps = num_data // num_per_chunk

                print("num_data = {0:d}, num_timestpes = {1:d}".format(num_data, num_timesteps))
        h5file.close()

# Continue by creating an adios2 file
adios = adios2.ADIOS(adios2.DebugON)
bpIO = adios.DeclareIO("KSTAR_018431")
bpIO.SetEngine("BP4")
bpWriter = bpIO.Open("KSTAR.bp", adios2.Mode.Write)

chlist_test = chlist_g[:2] + chlist_h[:2] + chlist_l[:2]


# Add all fields from the three H5 files to our bp file.
# Do this one file at a time
bp_var_list_g = []
bp_var_list_h = []
bp_var_list_l = []

# Add all the channels as variables to the bp file
dummy_arr = np.zeros(num_per_chunk, dtype=np.float64)
dummy_shape = [num_per_chunk]
dummy_start = [0]
dummy_count = [num_per_chunk]


for chlist, bp_var_list in zip([chlist_g, chlist_h, chlist_l],
                               [bp_var_list_g, bp_var_list_h, bp_var_list_l]):

    for ch_name in chlist:
        var = bpIO.DefineVariable(ch_name, dummy_arr, dummy_shape, dummy_start, dummy_count, adios2.ConstantDims)
        bp_var_list.append(var)


for tstep in range(num_timesteps):
    t0 = tstep * num_per_chunk
    t1 = (tstep + 1) * num_per_chunk

    print("Time step {0:d}: {1:d} - {2:d}".format(tstep, t0, t1))

    bpWriter.BeginStep()

    for bp_var_list, h5_fname, ch_list in zip([bp_var_list_g, bp_var_list_h, bp_var_list_l], 
                                              [h5fname_g, h5fname_h, h5fname_l],
                                              [chlist_g, chlist_h, chlist_l]): 
#        print(ch_list[0], bp_var_list[0].Name(), h5_fname)

        with h5py.File(h5_fname) as h5file:
            for ch, bp_var in zip(ch_list, bp_var_list):
                h5data = h5file["/ECEI/{0:s}/Voltage".format(ch)][t0:t1].astype(np.float64)
                #print(h5data.shape, h5data.dtype, dummy_arr.shape, dummy_arr.dtype)
                bpWriter.Put(bp_var, h5data)# 
    bpWriter.EndStep()


#bpWriter.close()

