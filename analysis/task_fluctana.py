# Coding: UTF-8 -*-

import numpy as np

# We need to add the projects directory to the pythonpath to accomodate dask
#import sys
#sys.path.append("/global/homes/r/rkube/repos/delta")
#from diagnostics.ecei_channel_layout import ecei_channel_positions

from analysis.channels import channel, channel_list
from analysis.fluctana import FluctAna
from analysis.kstarecei import KstarEcei
from analysis import specs as sp


#def test_cross_power(Dlist, done=0, dtwo=1, done_subset=None, dtwo_subset=None):
def test_cross_power(args):
    """Calculates cross-power for channels in a Dlist object
    Inputs:
    =======
    Dlist: kstar_ecei object
    channel_list: channel_list that labels the data in Dlist


    Outputs:
    ========
    # IN : data number one (ref), data number two (cmp), etc
    # OUT : x-axis (ax), y-axis (val)

    """
    Dlist, channel_list = args

    print("Dlist = ", Dlist)
    print("Channel_list = ", channel_list)


    for ch in channel_list:
        print(ch.to_str())

    # #Dlist[dtwo].vkind = 'cross_power'

    # if done_subset is not None: 
    #     num_refs = len(done_subset)
    # else:
    #     rnum = len(Dlist[done].data)  # number of ref channels
    #     done_subset = range(rnum)

    # if dtwo_subset is not None:
    #     num_chs = len(dtwo_subset)
    # else:
    #     num_chs = len(Dlist[dtwo].data)  # number of cmp channels
    #     dtwo_subset = range(cnum)

    # # reference channel names
    # Dlist[dtwo].rname = []

    # # value dimension
    # Dlist[dtwo].val = np.zeros((cnum, len(Dlist[dtwo].ax)))

    # # calculation loop for multi channels
    # for c in range(cnum):
    #     # reference channel number
    #     if num_refs == 1:
    #         Dlist[dtwo].rname.append(Dlist[done].clist[done_subset[0]])
    #         XX = Dlist[done].spdata[done_subset[0],:,:]
    #     else:
    #         Dlist[dtwo].rname.append(Dlist[done].clist[done_subset[c]])
    #         XX = Dlist[done].spdata[done_subset[c],:,:]

    #     YY = Dlist[dtwo].spdata[dtwo_subset[c],:,:]

    #     if Dlist[dtwo].ax[1] < 0: # full range
    #         Dlist[dtwo].val[c,:] = sp.cross_power(XX, YY, Dlist[dtwo].win_factor)
    #     else: # half
    #         Dlist[dtwo].val[c,:] = 2*sp.cross_power(XX, YY, Dlist[dtwo].win_factor)  


class task_fluctana():
    """A wrapper around FluctAna objects to be submitted to dask worker clients"""
    def __init__(self, shotnr, config, ecei_config):
        """
        Defines a list of fluctana tasks

        Input:
        ======
        shotnr: int, Shot number
        config: Dictionary 
        ecei_config, passed to KstarEcei
        """  

        self.ch_list = channel_list(channel.from_str(config["channels"][0]),
                                    channel.from_str(config["channels"][-1]))
        self.description = config["description"]
        self.analysis = config["analysis"]
        #self.fft_params = config["fft_params"]

        data_obj = KstarEcei(shotnr, [c.__str__() for c in self.ch_list], ecei_config)

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

        # if self.first_call:
        #     self.fluctana.fftbins(nfft=self.fft_params["nfft"], window=self.fft_params['window'], 
        #                           overlap=self.fft_params['overlap'], 
        #                           detrend=self.fft_params['detrend'], full=1)
        #     self.first_call = False

    def method(self, dask_client):
        """Creates a wrapper to call the appropriate analysis function"""
        dummy_data = np.random.uniform(0.0, 1.0, 100)
        return(dask_client.submit(print_path, dummy_data))


    def get_method(self):
        if self.analysis == "cross_power":
            print("cross_power it is")
            return test_cross_power

        return None

    def get_data(self):
        if self.analysis == "cross_power":
            return (self.fluctana.Dlist[0], self.ch_list)

        return None



# End of file task_dummy.py