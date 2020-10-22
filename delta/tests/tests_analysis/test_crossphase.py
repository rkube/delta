# Encoding: UTF-8 -*-

import numpy as np
from distributed import Client, progress
import dask.array as da
dask_client = Client(scheduler_file="/global/cscratch1/sd/rkube/scheduler.json")


with np.load("../dask_fft_data_s0000.npz") as df:
    num_channels, num_fft = df["fft_data"].shape
    print(num_channels, num_fft)

    fft_data = da.from_array(df["fft_data"], chunks=(1, num_fft))
    dask_client.persist(fft_data)


# Calculate the crosspower using the array interface
res1 = (fft_data[:2,:] * fft_data[-2:,:].conj()).mean(axis=1)
print("type res1 = ", type(res1))
res2 = da.arctan2(res1.real, res1.imag).real
print("type res2 = ", type(res2))
print("result res2 = ", res2.compute())


# Calculate the crosspower using the distributed interface
def cross_phase(ft_data, ch1, ch2):


    _tmp1 = (ft_data[ch1, :] * ft_data[ch2, :].conj()).mean().compute()
    print("** crosspower: type(tmp1) =", type(_tmp1))
    _tmp2 = np.arctan2(_tmp1.real, _tmp1.imag).real
    #_tmp2 = _tmp1.real + _tmp1.imag

    return(_tmp2)

res_d = dask_client.submit(cross_phase, fft_data, 1, 6)
print("type resd = ", type(res_d))
print("results resd = ", dask_client.gather(res_d))


# End of file test_crossphase.py