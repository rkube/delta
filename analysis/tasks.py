# coding: UTF-8 -*-
#from analysis.spectral import power_spectrum

from scipy.signal import welch


class analysis_task():
    """This is a simple task object that can be dispatched to dask workers.
    It stores data on which to perform an analysis in self.data
    It stores a definition of the analysis to be performed in self.task_name
    It stores parameters for the self.task_name in self.kw_dict
    """

    def __init__(self, channel_list, task_name, kw_dict):
        """Initialize the object with a fixed channel list, a fixed name of the analysis to be performed
        and a fixed set of parameters for the analysis routine

        Inputs:
        =======
        channel_list: list of strings, defines the name of the channels. This should probably match the
                      name of the channels in the BP file.
        task_name: string, defines the name of the analysis to be performed
        kw_dict: dict, is passed to the analysis routine
        """

        self.channel_list = channel_list
        self.task_name = task_name
        self.kw_dict = kw_dict

        self.data = {}

    
    def update_data(self, ch_data, ch_name):
        """Updates the data for a given channel name.

        Inputs:
        =======
        ch_data: ndarray, data with which to update the channel data
        ch_name: string, name of the channel to be updated.

        Returns:
        ========
        None
        """
        assert(ch_name in self.channel_list)
        self.data[ch_name] = ch_data


    def calculate(self, dask_client):
        """Performs the analysis.
        Input:
        ======
        client: dask client

        Output:
        =======
        future: dask future that holds the result of the analysis.
        """

        self.future = None

        if self.task_name == "power_spectrum":
            ch0 = self.channel_list[0]
            self.future = dask_client.submit(welch, self.data[ch0])

        return(self.future)


# End of file analysis_package.py