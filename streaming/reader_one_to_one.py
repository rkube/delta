#-*- Coding: UTF-8 -*-


#from mpi4py import MPI
import adios2
import logging

import numpy as np
from analysis.channels import channel, channel_range


class reader_base():
    def __init__(self, shotnr, ecei_cfg):
        self.adios = adios2.ADIOS()
        self.shotnr = shotnr
        self.IO = self.adios.DeclareIO("KSTAR_18431")
        # Keeps track of the past chunk sizes. This allows to construct a dummy time base
        self.chunk_sizes = []
        
        # Store configuration of the ECEI diagnostic
        self.ecei_cfg = ecei_cfg

        # If false, indicates that raw data is returned.
        # If true, indicates that normalized data is returned.
        # This flag is set in Get()
        self.is_data_normalized = False

        # Defines the time where we take the offset
        self.tnorm = ecei_cfg["t_norm"]


    def Open(self, datapath):
        """Opens a new channel"""
        from os.path import join

        self.channel_name = join(datapath, "KSTAR.bp".format(self.shotnr))
        if self.reader is None:
            self.reader = self.IO.Open(self.channel_name, adios2.Mode.Read)

    def BeginStep(self):
        """Wrapper for reader.BeginStep()"""
        logging.info("Reader::BeginStep")
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


    def InquireVariable(self, varname):
        """Wrapper for IO.InquireVariable"""
        res = self.IO.InquireVariable(varname)
        return(res)


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


    
    def Get(self, channels=None, save=False):
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
        channels: Either None, a string or a list of channels. Attempt to read all ECEI channels
                  if channels is None
        
        Returns:
        ========
        io_array: numpy ndarray containing data of the current step
        """


        print("Reader::Get")

        if (isinstance(channels, channel_range)):
            data_list = []
            for c in channels:
                var = self.IO.InquireVariable("ECEI_" + str(c))
                io_array = np.zeros(np.prod(var.Shape()), dtype=np.float64)
                self.reader.Get(var, io_array, adios2.Mode.Sync)
                data_list.append(io_array)

            # Append size of current chunk to chunk sizes
            self.chunk_sizes.append(data_list[0].size)
            io_array = np.array(data_list) * 1e-4

        elif isinstance(channels, type(None)):
            data_list = []
            print("Reader::Get*** Default reading channels L0101-L2408. Step no. {0:d}".format(self.CurrentStep()))
            clist = channel_range(channel("L", 1, 1), channel("L", 24, 8))

            # Inquire the data of each channel from the ADIOS IO
            for c in clist:
                var = self.IO.InquireVariable("ECEI_" + str(c))
                io_array = np.zeros(np.prod(var.Shape()), dtype=np.float64)
                self.reader.Get(var, io_array, adios2.Mode.Sync)
                data_list.append(io_array)

            # Append size of current chunk to chunk sizes
            self.chunk_sizes.append(data_list[0].size)
            io_array = np.array(data_list) * 1e-4
            
        # If the normalization offset hasn't been calculated yet see if we have the
        # correct data to do so in the current chunk
        if self.is_data_normalized == False:
            # Generate the timebase for the current step
            tb = self.gen_timebase()
            # Calculate indices where we calculate the normalization offset from
            tnorm_idx = (tb > self.tnorm[0]) & (tb < self.tnorm[1])
            print("*** Reader: I found {0:d} indices where to normalize".format(tnorm_idx.sum()), ", tnorm = ", self.tnorm)
            # Calculate normalization offset if we have enough indices
            if(tnorm_idx.sum() > 100):
                self.offset_lvl = np.median(io_array[:, tnorm_idx], axis=1, keepdims=True)
                self.offset_std = io_array[:, tnorm_idx].std(axis=1)
                self.is_data_normalized = True

                if save:
                    np.savez("test_data/offset_lvl.npz", offset_lvl = self.offset_lvl)

        if self.is_data_normalized:
            print("*** Reader:Get: io_array.shape = ", io_array.shape)
            if save:
                np.savez("test_data/io_array_s{0:04d}.npz".format(self.CurrentStep()), io_array=io_array)
            io_array = io_array - self.offset_lvl
            io_array = io_array / io_array.mean(axis=1, keepdims=True) - 1.0
            if save:
                np.savez("test_data/io_array_tr_s{0:04d}.npz".format(self.CurrentStep()), io_array=io_array)


        return io_array


class reader_bpfile(reader_base):
    def __init__(self, shotnr, ecei_cfg):
        super().__init__(shotnr, ecei_cfg)
        self.IO.SetEngine("BP4")
        self.reader = None


# End of file reader_one_to_one.py