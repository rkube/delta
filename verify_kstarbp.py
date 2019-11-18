import adios2
import numpy as np
import h5py

varname = "L0101"


adios = adios2.ADIOS()
IO = adios.DeclareIO("reader")
IO.SetEngine("BP4")

#bpStream = IO.Open("KSTAR.bp", adios2.Mode.Read)
bpStream = IO.Open("/global/cscratch1/sd/rkube/KSTAR/kstar_streaming/KSTAR_018431.bp", adios2.Mode.Read)
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
    #print(h5data)

    print(varname, "*** BP: min = {0:6f}, max = {1:6f}, mean = {2:6f}".format(res.min(), res.max(), res.mean()))
    print(varname, "*** H5: min = {0:6f}, max = {1:6f}, mean = {2:6f}".format(h5data.min(), h5data.max(), h5data.mean()))

    if(step > 10):
        break

df_h5.close()


