# -*- Encoding: UTF-8


"""Verifies that data in bp-file is correct.

Compares the output of the data file written by create_ecei_bpfile.py to
the original data in the HDF5 file.
"""


# import sys
import numpy as np
import h5py

# sys.path.append("/global/homes/r/rkube/software/adios2-current/lib/python3.8/site-packages")
import adios2

varname = "L0101"


adios = adios2.ADIOS()
IO = adios.DeclareIO("reader")
IO.SetEngine("BP4")

# bpStream = IO.Open("KSTAR.bp", adios2.Mode.Read)
bpStream = IO.Open("/global/cscratch1/sd/rkube/KSTAR/kstar_streaming/KSTAR_018431.bp",
                   adios2.Mode.Read)
df_h5 = h5py.File("/global/cscratch1/sd/rkube/KSTAR/kstar_streaming/018431/ECEI.018431.LFS.h5")

ds = 10000
res = np.zeros(10000, dtype=np.float64)


for step in range(bpStream.Steps()):
    print("***Step: {0:d}".format(step))
    bpStream.BeginStep()
    var = IO.InquireVariable("ECEI_{0:s}".format(varname))
    L2408 = bpStream.Get(var, res, adios2.Mode.Sync)
    bpStream.EndStep()

    h5data = df_h5["/ECEI/ECEI_{0:s}/Voltage".format(varname)][(step) * ds:(step + 1) * ds]
    # print(h5data)

    print(varname,
          "*** BP: min = {res.min():6f}, max = {res.max():6f}, mean = {res.mean():6f}")
    print(varname,
          "*** H5: min = {h5data.min():6f}, max = {h5data.max():6f}, mean = {h5data.mean():6f}")

    if(step > 10):
        break

df_h5.close()

# End of file verify_kstarbp.py
