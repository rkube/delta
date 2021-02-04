# -*- Encoding: UTF-8 -*-

import sys
import os
sys.path.append("/global/homes/r/rkube/repos/delta/delta")

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import json

from scipy.signal import stft

from data_models.kstar_ecei import ecei_chunk
from analysis.task_list import tasklist
from data_models.timebase import timebase_streaming
from azure.storage.blob import BlockBlobService


# Create the BlockBlobService that is used to call the Blob service for the storage account
blob_service_client = BlockBlobService(account_name="deltafiles")
# Files are stored in testfiles
container_name = 'testfiles'
local_file_name = "ecei_22289_cross_phase.npz"

# Download the blob(s).
# Add '_DOWNLOADED' as prefix to '.txt' so you can see both files in Documents.
full_path_to_file = os.path.join(os.getcwd(), str.replace(local_file_name, '.npz', '_DOWNLOADED.npz'))
blob_service_client.get_blob_to_path(container_name, local_file_name, full_path_to_file)

with np.load(full_path_to_file) as df:
    print(df.files)
    clist1 = df["clist1"]
    clist2 = df["clist2"]
    cross_phase_fa = np.squeeze(df["cross_phase_data"])
    dname = df["dname"]
    flimits = df["flimits"]
    an_name = df["an_name"]
    trange = df["trange"]
    context = df["context"]
    data_dlist0 = np.squeeze(df["data_dlist0"])
    data_dlist1 = np.squeeze(df["data_dlist1"])

# Clean up the temp file
os.remove(full_path_to_file)


# Run analysis using Delta routines
cfg_an = json.loads(""" {
    "storage": {"backend": "numpy", "basedir": "/global/homes/r/rkube"},
    "analysis": 
    {"crossphase": {
      "channel_chunk_size": 32768, 
      "ref_channels": [1, 1, 1, 2], 
      "cmp_channels": [1, 2, 1, 2]
    }} 
    }""")

e = ThreadPoolExecutor(max_workers=2)
tb = timebase_streaming(2.716, 2.736, 5e5, 10000, 0)

dl1_ft = stft(data_dlist0, fs=5e5, nperseg=512, noverlap=256, detrend="constant", window="hann", return_onesided=True)[2]
dl0_ft = stft(data_dlist1, fs=5e5, nperseg=512, noverlap=256, detrend="constant", window="hann", return_onesided=True)[2]

all_data = np.stack([dl0_ft, dl1_ft])
print(all_data.shape)

my_chunk = ecei_chunk(all_data, tb, None, num_v=2, num_h=1)
my_tasklist = tasklist(e, cfg_an)
my_tasklist.execute(my_chunk)