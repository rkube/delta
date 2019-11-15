# Coding: UTF-8 -*-

import numpy as np

# We need to add the projects directory to the pythonpath to accomodate dask
#import sys
#sys.path.append("/global/homes/r/rkube/repos/delta")
#from diagnostics.ecei_channel_layout import ecei_channel_positions

from analysis.channels import channel, channel_list
from analysis.fluctana import FluctData, FluctAna
from analysis.kstarecei import KstarEcei


# Defines rpos (in meters), zpos (in meters) and apos (in meters) for
# the individual channels of the ECEI diagnostic
ecei_channel_positions = {"L2403": (4.3, 4.3, 4.3),
                          "L2406": (4.6, 4.6, 4.6)}

class task_fluctana():
    """A wrapper around FluctAna objects to be submitted to dask worker clients"""
    def __init__(self, shotnr, config):
        """
        Defines a list of fluctana tasks

        Input:
        ======
        shotnr: int, Shot number
        config: Dictionary 
        """  

        self.ch_list = channel_list(channel.from_str(config["channels"][0]),
                                    channel.from_str(config["channels"][-1]))
        self.description = config["description"]
        self.analysis_list = config["analysis_list"]
        self.fft_params = config["fft_params"]

        
        data_obj = KstarEcei(shotnr, [c.to_str() for c in self.ch_list], config)

        print("Init...", data_obj.tt)
        self.fluctana = FluctAna()
        self.fluctana.Dlist.append(data_obj)

        # Flag that is set to False after first call to update_data
        self.first_call = True


    def update_data(self, ch_data, trange):
        """Updates data for all channels.
        Inputs:
        =======
        ch_data: ndarray, shape(M, N) where M is the number of channels and N the number of data points
        trange: ndarray, shape(1,N) where N is the number of data points.

        Returns:
        ========
        None

        """
        
        # First we update the time range
        print("Updating. trnage = ", trange, trange.shape)
        self.fluctana.Dlist[0].time, _, _, _, _ = self.fluctana.Dlist[0].time_base(trange)
        print("Updated time: ", self.fluctana.Dlist[0].time)
        # Now we update the data
        self.fluctana.Dlist[0].data = ch_data

        if self.first_call:

            self.fluctana.fftbins(nfft=self.fft_params["nfft"], window=self.fft_params['window'], 
                                  overlap=self.fft_params['overlap'], 
                                  detrend=self.fft_params['detrend'], full=1)
            self.first_call = False

    def create_task_object(self):
        # Create a dummy time range
        #dummy_time = np.arange(self.data[next(iter(self.data))].size)
        dummy_time = np.arange(10000)

        # Adapt data format to FluctData
        rpos = [pos[0] for pos in [ecei_channel_positions[c] for c in ["L2403", "L2406"]]]
        zpos = [pos[1] for pos in [ecei_channel_positions[c] for c in ["L2403", "L2406"]]]
        apos = [pos[2] for pos in [ecei_channel_positions[c] for c in ["L2403", "L2406"]]]

        # Flatten the channel data dictionary into a numpy array.
        # dim0: channel_list[0], channel_list[1]...
        # dim1: time, see dummy_time above
        ch_data = np.zeros((len(self.channel_list), dummy_time.size), dtype=np.float64)
        #for idx, kv_tuple in enumerate(self.data.items()):
        #    ch_data[idx, :] = kv_tuple[1]

        self.fluct_data = FluctData(18431, self.channel_list, dummy_time, ch_data, rpos, zpos, apos)


    def method(self):
        fun = lambda x: x

        return(fun)

#    def data(self):
#        return np.random.uniform(0.0, 1.0, 10)


# End of file task_dummy.py