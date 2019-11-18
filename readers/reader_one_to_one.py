#-*- Coding: UTF-8 -*-

import adios2
import numpy as np
from analysis.channels import channel, channel_list


class reader_base():
    def __init__(self, shotnr):
        self.adios = adios2.ADIOS()
        self.shotnr = shotnr
        self.IO = self.adios.DeclareIO("KSTAR_18431")


    def Open(self, datapath):
        """Opens a new channel"""
        from os.path import join

        self.channel_name = join(datapath, "KSTAR.bp".format(self.shotnr))
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


    def InquireVariable(self, varname):
        """Wrapper for IO.InquireVariable"""
        res = self.IO.InquireVariable(varname)
        return(res)

    
    def Get(self, varname):
        """Get data from varname at current step.
        Inputs:
        =======
        varname: Either a string or a list of channels
        
        Returns:
        ========
        io_array: numpy ndarray for the data
        """

        if (isinstance(varname, str)):
            var = self.IO.InquireVariable("ECEI_" + varname)
            io_array = np.zeros(np.prod(var.Shape()), dtype=np.float64)
            self.reader.Get(var, io_array, adios2.Mode.Sync)
            return(io_array)


        elif (isinstance(varname, channel_list)):
            data_list = []
            for v in varname:
                var = self.IO.InquireVariable("ECEI_" + v.to_str())
                io_array = np.zeros(np.prod(var.Shape()), dtype=np.float64)
                self.reader.Get(var, io_array, adios2.Mode.Sync)
                data_list.append(io_array)

            return np.array(data_list)

        return None


class reader_bpfile(reader_base):
    def __init__(self, shotnr):
        super().__init__(shotnr)
        self.IO.SetEngine("BP4")
        self.reader = None


# End of file reader_one_to_one.py