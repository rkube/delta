# -*- Encoding: UTF-8 -*-

from mpi4py import MPI 
import numpy as np
import adios2

from processors.readers import reader_dataman


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


my_reader = reader_dataman(18431, 11)
my_reader.Open()

step = 0
print("Waiting")

while(True):
#for i in range(10):
    stepStatus = my_reader.BeginStep()
    #print(stepStatus)
    if stepStatus == adios2.StepStatus.OK:
        #var = dataman_IO.InquireVariable("floats")
        #shape = var.Shape()
        #io_array = np.zeros(np.prod(shape), dtype=np.float)
        #reader.Get(var, io_array, adios2.Mode.Sync)
        channel_data = my_reader.get_data("floats")
        #currentStep = reader.CurrentStep()
        my_reader.EndStep()
        #print("rank {0:d}: Step".format(rank), reader.CurrentStep(), ", io_array = ", io_array)
    else:
        print(">>> receiver {0:d}: End of stream".format(rank))
        break

    # Recover channel data 
    #channel_data = channel_data.reshape((num_channels, channel_data.size // num_channels))

    print (f">>> Step begins ... {step}")
    ## jyc: this is just for testing. This is a place to run analysis if we want.
    ##executor.submit(perform_analysis, channel_data, step)

    ## Save data in a queue so that a workder thead will fetch and save concurrently.
    #queue_list[step%num_analysis].put((channel_data, step))
    step += 1


# End of file processor_simple.py