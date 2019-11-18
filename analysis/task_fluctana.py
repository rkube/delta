# Coding: UTF-8 -*-

import numpy as np

# We need to add the projects directory to the pythonpath to accomodate dask
import sys
sys.path.append("/global/homes/r/rkube/repos/delta")
#from diagnostics.ecei_channel_layout import ecei_channel_positions

from analysis.channels import channel, channel_list
from analysis.fluctana import FluctAna
from analysis.kstarecei import KstarEcei


# Defines rpos (in meters), zpos (in meters) and apos (in meters) for
# the individual channels of the ECEI diagnostic
#ecei_channel_positions = {"L2403": (4.3, 4.3, 4.3),
#                          "L2406": (4.6, 4.6, 4.6)}


def test_cross_power(Dlist, done=0, dtwo=1, done_subset=None, dtwo_subset=None):
    """Calculates cross-power for channels in a Dlist object
    Inputs:
    =======


    Outputs:
    ========
    # IN : data number one (ref), data number two (cmp), etc
    # OUT : x-axis (ax), y-axis (val)

    """

    #Dlist[dtwo].vkind = 'cross_power'

    if done_subset is not None: 
        rnum = len(done_subset)
    else:
        rnum = len(Dlist[done].data)  # number of ref channels
        done_subset = range(rnum)

    if dtwo_subset is not None:
        cnum = len(dtwo_subset)
    else:
        cnum = len(Dlist[dtwo].data)  # number of cmp channels
        dtwo_subset = range(cnum)

    # reference channel names
    Dlist[dtwo].rname = []

    # value dimension
    Dlist[dtwo].val = np.zeros((cnum, len(Dlist[dtwo].ax)))

    # calculation loop for multi channels
    for c in range(cnum):
        # reference channel number
        if rnum == 1:
            Dlist[dtwo].rname.append(Dlist[done].clist[done_subset[0]])
            XX = Dlist[done].spdata[done_subset[0],:,:]
        else:
            Dlist[dtwo].rname.append(Dlist[done].clist[done_subset[c]])
            XX = Dlist[done].spdata[done_subset[c],:,:]

        YY = Dlist[dtwo].spdata[dtwo_subset[c],:,:]

        if Dlist[dtwo].ax[1] < 0: # full range
            Dlist[dtwo].val[c,:] = sp.cross_power(XX, YY, Dlist[dtwo].win_factor)
        else: # half
            Dlist[dtwo].val[c,:] = 2*sp.cross_power(XX, YY, Dlist[dtwo].win_factor)  


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
        self.analysis = config["analysis"]
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
        print("Updating. trange.shape = ", trange.shape)
        self.fluctana.Dlist[0].time, _, _, _, _ = self.fluctana.Dlist[0].time_base(trange)
        # Now we update the data
        self.fluctana.Dlist[0].data = ch_data
        self.fluctana.Dlist[0].trange = trange
        self.fluctana.list_data()

        if self.first_call:
            self.fluctana.fftbins(nfft=self.fft_params["nfft"], window=self.fft_params['window'], 
                                  overlap=self.fft_params['overlap'], 
                                  detrend=self.fft_params['detrend'], full=1)
            self.first_call = False

    def method(self, dask_client):
        """Creates a wrapper to call the appropriate analysis function"""
        dummy_data = np.random.uniform(0.0, 1.0, 100)

        #if self.analysis == "cross_power":
            #fun = lambda A: A.cross_power
        #return(dask_client.submit(test_analysis, dummy_data))
        return(dask_client.submit(print_path, dummy_data))
        #return(None)


    def get_method(self):
        #return np.mean
        my_mean = test_cross_power
        return my_mean

    def get_data(self):
        #return np.random.uniform(0.0, 1.0, 100)
        return self.fluctana.Dlist[0].data



# End of file task_dummy.py