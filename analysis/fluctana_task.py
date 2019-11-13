# coding: UTF-8 -*-

from fluctana.dummy import dummy

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


    def update_data(self, ch_data, ch_name):
        """Updates data for a single channel.
        """

        assert(ch_name in self.channel_list)
        self.data[ch_name] = ch_data


    def create_fluctdata_object(self):
        self.fluct_data = FluctData(0, self.channel_list, )



# End of file fluctana_task.py