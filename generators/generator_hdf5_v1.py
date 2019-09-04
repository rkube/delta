# -*- coding: UTF-8 -*-

from kafka import  KafkaProducer
import h5py
import pickle
import time


"""
Generates batches of ECEI data.
"""

shotnr = 18431
data_path = "/global/cscratch1/sd/rkube/KSTAR/kstar_streaming/"
df_fname = "{0:s}/{1:06d}/ECEI.{1:06d}.HFS.h5".format(data_path, shotnr, shotnr)


# Specify the channel we are streaming
channel_name = "H2202"

# Hard-code the total number of data points
data_pts = int(5e6)
# Hard-code the total number of data batches we send over the channel
batches = int(1e2)
# Calculate the number of data points per data packet
data_per_batch = data_pts // batches

producer = KafkaProducer()

with h5py.File(df_fname, "r") as df:
    for bb in range(batches):
        dset_name = "/ECEI/ECEI_{0:s}/Voltage".format(channel_name)
        data = df[dset_name][bb *data_per_batch:(bb + 1) * data_per_batch]
        print(bb, data.shape, type(data))
        producer.send(channel_name, pickle.dumps(data))
        time.sleep(1e-1)




def read_attrs(df_fname):
    """Read attributes from a KSTAR ECEI hdf5 file. Return attributes as a dictionary

    Parameters
    ----------
    df_fname : str
        input file name

    Returns
    -------
        attrs : dict
            attributes of the HDF5 file
    """

    with h5py.File(df_fname, "r") as df:
        dset = df["/ECEI"]
        attrs = dict(dset.attrs)
        attrs["toff"] = attrs["TriggerTime"][0] + 1e-3
        
        try:
            attrs["Mode"] = attrs["Mode"].decode()

            if attrs["Mode"] == "O":
                hn = 1
            elif attrs["Mode"] == "X":
                hn = 2
        except:
            attrs["Mode"] = "X"
            hn = 2

        attrs["SampleRate"][0] = attrs["SampleRate"][0] * 1e3
        attrs["TFcurrent"] = attrs["TFcurrent"] * 0.995556

    return(attrs)


def gen_timebase(tstart, tend):
    """Generates a time base for ECEI data

    Parmeters
    ---------
    tstart : float
        start time
    tend : float
        end time

    Returns 
    -------
        
    """

    # Define some shortcuts
    tt = attrs["TriggerTime"]
    toff = attrs["toff"]
    fs = attrs["SampleRate"][0]





    return None



# End of file hdf5_generator.py