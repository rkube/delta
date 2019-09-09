#-*- coding: UTF-8 -*-

from mpi4py import MPI
import numpy as np 
import adios2


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

channel = 2202

transport_params = {"IPAddress": "127.0.0.1",
                    "Port": "12306"}


adios = adios2.ADIOS(comm)
dataman_IO = adios.DeclareIO("ECEI_H{0:4d}".format(channel))
dataman_IO.SetEngine("DataMan")
dataman_IO.SetParameters(transport_params)
dataman_IO.AddTransport("WAN", transport_params)

datamanReader = dataman_IO.Open("stream", adios2.Mode.Read)

while(True):
    stepStatus = datamanReader.BeginStep()
    if stepStatus == adios2.StepStatus.OK:
        var = dataman_IO.InquireVariable("floats")
        shape = var.Shape()
        io_array = np.zeros(np.prod(shape), dtype=np.float)

        datamanReader.Get(var, io_array, adios2.Mode.Sync)
        currentStep = datamanReader.CurrentStep()
        datamanReader.EndStep()
        print("Step", currentStep, ", shape = ", shape, ", io_array = ", io_array)
    else:
        print("End of stream")
        break

datamanReader.Close()


# End of file processor_adios2.