# coding: UTF-8 -*-

import numpy as np
#from fluctana.dummy import dummy
from fluctana.fluctana import FluctData, FluctAna
from diagnostics.ecei_channel_layout import ecei_channel_positions



class fluctana_task():
    """A prototype for dispatching tasks using kstar fluctana objects.
    """

    def __init__(self, channel_list, description, analysis_list, param_list):
        """
        Defines a list of fluctana tasks

        Input:
        ======
        channel_list: list of strings, channels to be analyzed
        description: string, a description of this task
        analysis_list: string, list of FluctAna methods to be executed
        param_list: dict, parameters for FluctAna methods
        """

        self.channel_list = channel_list
        self.description = description
        self.analysis_list = analysis_list
        self.param_list = param_list

        self.data = {}
        self.fluctana = None
        self.futures = None


    def update_data(self, ch_data, ch_name):
        """Updates data for a single channel.
        """

        assert(ch_name in self.channel_list)
        self.data[ch_name] = ch_data


    def create_fluctdata_object(self):
        # Create a dummy time range
        dummy_time = np.arange(self.data[next(iter(self.data))].size)

        # Adapt data format to FluctData
        rpos = [pos[0] for pos in [ecei_channel_positions[c] for c in ["L2403", "L2406"]]]
        zpos = [pos[1] for pos in [ecei_channel_positions[c] for c in ["L2403", "L2406"]]]
        apos = [pos[2] for pos in [ecei_channel_positions[c] for c in ["L2403", "L2406"]]]

        # Flatten the channel data dictionary into a numpy array.
        # dim0: channel_list[0], channel_list[1]...
        # dim1: time, see dummy_time above
        ch_data = np.zeros((len(self.data.keys()), dummy_time.size), dtype=np.float64)
        for idx, kv_tuple in enumerate(self.data.items()):
            ch_data[idx, :] = kv_tuple[1]

        self.fluct_data = FluctData(18431, self.channel_list, dummy_time, ch_data, rpos, zpos, apos)


    def dispatch_analysis_task(self, dask_client):
        """Dispatches all analysis tasks to the workers.

        Input:
        ======
        client: dask client

        Output:
        =======
        future: dask future that holds the results of the analysis
        """

        # TODO: This deletes all previous futures. We should probably check if this
        # needs some extra checking,
        self.futures = []

        fluct_ana = FluctAna()
        fluct_ana.add_data(self.fluct_data, np.arange(10000), verbose=0)

        self.future = dask_client.submit(fluct_ana.fftbins, self.param_list)

#   A.fftbins(nfft=cfg['nfft'],window=cfg['window'],
#              overlap=cfg['overlap'],detrend=cfg['detrend'],full=1)

        #for analysis in self.analysis_list:





# End of file fluctana_task.py