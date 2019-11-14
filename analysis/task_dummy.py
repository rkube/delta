# Coding: UTF-8 -*-

import numpy as np

class task_dummy():
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


    def create_task_object(self):
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


    def method(self):
        fun = lambda x: x

        return(fun)

#    def data(self):
#        return np.random.uniform(0.0, 1.0, 10)


# End of file task_dummy.py