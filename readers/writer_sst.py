# Encoding: UTF-8 -*-

import numpy as np
import adios2


class writer_sst():
    """Defines an sst writer for a single variable"""
    def __init__(self, ch_name, varname, varshape, dtype, a2=None):
        """ Initializes the SST writer

        Input:
        ======
        a2: adios2 object
        ch_name, str: Name of the channel
        varname, str: Name of the variable in the SST channel
        varshape, tuple of ints: Shape of the variable in the SST channel
        dtype, type: Type of the data we are writing
        """

        if a2 == None:
            self.adios = adios2.ADIOS()
        else:
            self.adios = a2

        self.sstIO = self.adios.DeclareIO(ch_name)
        self.sstIO.SetEngine("SST")
        self.sstWriter = self.sstIO.Open(filename, adios2.Mode.Write)

        self.varname = varname
        self.varshape = varshape
        self.dtype = dtypee

        self.var = self.sstIO.DefineVariable(varname, shape=varshape,
                                             start=(0,),
                                             count=varshape,
                                             isConstantDims=True)


    def BeginStep(self):
        """Wrapper for adios2.writer.BeginStep()"""

        res = self.sstWriter.BeginStep()

        if res == adios2.StepStatus.OK:
            return True

        return False


    def EndStep(self):
        """Wrapper for adios2.writer.EndStep()"""
        res = self.sstWriter.EndStep()

        if res == adios2.StepStatus.OK:
            return True

        return False

    def Put(self, data):
        """Wrapper for adios2.writer.Put()"""


        assert(data.dtype == self.dtype)
        assert(data.shape == self.varshape)

        self.sstWriter.Put(self.varname, data)


        return None



# End of file writer.py