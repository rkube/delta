#-*- Coding: UTF-8 -*-

from mpi4py import MPI
import adios2
import logging
import json
from os.path import join

import numpy as np




from analysis.channels import channel, channel_range
from streaming.adios_helpers import gen_io_name, gen_channel_name_v2

"""
Author: Ralph Kube
"""


class reader_base():
    def __init__(self, cfg: dict):
        """Generates a reader for KSTAR ECEI data.

        Parameters:
        -----------
        cfg: delta config dictionary
        """

        comm  = MPI.COMM_SELF
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        # This should be MPI.COMM_SELF, not MPI.COMM_WORLD
        self.adios = adios2.ADIOS(MPI.COMM_SELF)
        self.logger = logging.getLogger("simple")

        self.shotnr = cfg["shotnr"]
        self.IO = self.adios.DeclareIO(gen_io_name(self.shotnr))
        # Keeps track of the past chunk sizes. This allows to construct a dummy time base
        self.chunk_sizes = []
        
        # Store configuration of the ECEI diagnostic
        self.ecei_cfg = cfg["ECEI_cfg"]

        # If false, indicates that raw data is returned.
        # If true, indicates that normalized data is returned.
        # This flag is set in Get()
        self.is_data_normalized = False

        # Defines the time where we take the offset
        self.tnorm = cfg["ECEI_cfg"]["t_norm"]

        self.reader = None
        # Generate a descriptive channel name
        chrg = channel_range.from_str(cfg["transport"]["channel_range"][self.rank])
        self.channel_name = gen_channel_name_v2(self.shotnr, chrg.to_str())
        self.logger.info(f"reader_base: channel_name =  {self.channel_name}")


    def Open(self):
        """Opens a new channel"""

        self.logger.info(f"Waiting to receive channel name {self.channel_name}")
        if self.reader is None:
            self.reader = self.IO.Open(self.channel_name, adios2.Mode.Read)

    def BeginStep(self):
        """Wrapper for reader.BeginStep()"""
        res = self.reader.BeginStep()
        if res == adios2.StepStatus.OK:
            return(True)
        return(False)


    def CurrentStep(self):
        """Wrapper for IO.CurrentStep()"""
        res = self.reader.CurrentStep()
        return(res)


    def EndStep(self):
        """Wrapper for reader.EndStep"""
        res = self.reader.EndStep()
        return(res)


    def InquireVariable(self, varname: str):
        """Wrapper for IO.InquireVariable"""
        res = self.IO.InquireVariable(varname)
        return(res)


    def get_attrs(self, attrsname: str):
        """Inquire json string `attrsname` from the opened stream"""

        attrs = self.IO.InquireAttribute(attrsname)
        return json.loads(attrs.DataString()[0])


    def gen_timebase(self):
        """Create a dummy time base for chunk last read."""

        # Unpack the trigger time, plasma time and end time from TriggerTime
        tt0, pl, tt1 = self.ecei_cfg["TriggerTime"]
        # Get the sampling frequency, in units of Hz
        fs = self.ecei_cfg["SampleRate"] * 1e3

        # The time base starts at t0. Assume samples are streaming in continuously with sampling frequency fs.
        # Offset of the first sample in current chunk
        offset = sum(self.chunk_sizes[:-1])
        t_sample = 1. / fs
        # Use integers in arange to avoid round-off errors. We want exactly chunk_sizes[-1] elements in
        # this array
        tb = np.arange(offset, offset + self.chunk_sizes[-1], dtype=np.float64)
        # Scale timebase with t_sample and shift to tt0
        tb = (tb * t_sample) + tt0        
        return(tb)

    
    def Get(self, ch_rg: channel_range, save: bool=False):
        """Get data from varname at current step.

        The ECEI data is usually normalized to a fixed offset, calculated using data 
        at the beginning of the stream.

        The time interval where the data we normalize to is taken from is given in ECEI_config, t_norm.
        As long as this data is not seen by the reader, raw data is returned.
        
        Once the data we normalize to is seen, the normalization values are calculated.
        After that, the data from the current and all subsequent chunks is normalized.
    
        The flag self.is_data_normalized is set to false if raw data is returned.
        It is set to true if normalized data is returned.


        Inputs:
        =======
        ch_rg: channel_range that describes which channels to inquire. This is used to generate
               a variable name which is inquired from the stream
             

        Returns:
        ========
        time_chunk: numpy ndarray containing data of the current step
        """


        # elif isinstance(channels, type(None)):
        self.logger.info(f"Reader::Get*** Default reading varname {ch_rg.to_str()}. Step no. {self.CurrentStep():d}")
        var = self.IO.InquireVariable(ch_rg.to_str())
        time_chunk = np.zeros(var.Shape(), dtype=np.float64)
        self.reader.Get(var, time_chunk, adios2.Mode.Sync)
        self.logger.info(f"Got data: {time_chunk.shape}")
        # Append size of current chunk to chunk sizes
        self.chunk_sizes.append(time_chunk.shape[1])
        time_chunk = time_chunk * 1e-4
            
        # If the normalization offset hasn't been calculated yet see if we have the
        # correct data to do so in the current chunk
        if self.is_data_normalized == False:
            # Generate the timebase for the current step
            tb = self.gen_timebase()
            # Calculate indices where we calculate the normalization offset from
            tnorm_idx = (tb > self.tnorm[0]) & (tb < self.tnorm[1])
            self.logger.debug(f"*** Reader: I found {tnorm_idx.sum():d} indices where to normalize, tnorm = {self.tnorm}")
            # Calculate normalization offset if we have enough indices
            if(tnorm_idx.sum() > 100):
                self.offset_lvl = np.median(time_chunk[:, tnorm_idx], axis=1, keepdims=True)
                self.offset_std = time_chunk[:, tnorm_idx].std(axis=1)
                self.is_data_normalized = True

                if save:
                    np.savez("test_data/offset_lvl.npz", offset_lvl = self.offset_lvl)

        if self.is_data_normalized:
            self.logger.debug(f"*** Reader:Get: time_chunk.shape = {time_chunk.shape}")
            if save:
                np.savez("test_data/time_chunk_s{0:04d}.npz".format(self.CurrentStep()), time_chunk=time_chunk)

            time_chunk = time_chunk - self.offset_lvl
            time_chunk = time_chunk / time_chunk.mean(axis=1, keepdims=True) - 1.0

            if save:
                np.savez(f"test_data/time_chunk_tr_s{self.CurrentStep():04d}.npz", time_chunk=time_chunk)

        return time_chunk


class reader_gen(reader_base):
    def __init__(self, cfg: dict):
        """Instantiates a reader.
           Control Adios method and params through cfg

        Parameters:
        -----------
        cfg : delta config dict
        """
        super().__init__(cfg)
        self.IO.SetEngine(cfg["transport"]["engine"])
        ## Set port number for DataMan
        if cfg["transport"]["engine"].lower() == "dataman":
            cfg["transport"]["params"].update(Port = str(int(cfg["transport"]["params"]["Port"]) + self.rank))
        self.IO.SetParameters(cfg["transport"]["params"])

# End of file 
