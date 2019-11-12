# coding: UTF-8 -*-
#from analysis.spectral import power_spectrum

from scipy.signal import welch


class analysis_task():
    def __init__(self, channel_list, task_name, kw_dict):
        """Defines a list of analysis tasks to be performed. Data is loaded dynamically."""
        self.channel_list = channel_list
        self.task_name = task_name
        self.kw_dict = kw_dict

        self.data = {}

    
    def update_data(self, arr, ch_name):
        """Updates the data for a given channel"""
        assert(ch_name in self.channel_list)
        self.data[ch_name] = arr


    def calculate(self, dask_client):
        """Performs the analysis.
        Input:
        ======
        client: dask client

        Output:
        =======
        result: dask future that gives us the result
        """

        self.future = None

        if self.task_name == "power_spectrum":
            ch0 = self.channel_list[0]
            self.future = dask_client.submit(welch, self.data[ch0])

        return(self.future)


# End of file analysis_package.py