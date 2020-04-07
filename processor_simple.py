# -*- Encoding: UTF-8 -*-

from mpi4py import MPI 
import numpy as np
import adios2
import argparse
import json
import yaml
import logging, logging.config

from analysis.channels import channel_range
from streaming.readers import reader_dataman


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

parser = argparse.ArgumentParser(description="Send KSTAR data using ADIOS2")
parser.add_argument('--config', type=str, help='Lists the configuration file', default='configs/test_generator.json')
args = parser.parse_args()

with open(args.config, "r") as df:
    cfg = json.load(df)

with open('configs/logger.yaml', 'r') as f:
    log_cfg = yaml.safe_load(f.read())
logging.config.dictConfig(log_cfg)

logger = logging.getLogger("simple")

logger.info(f"{cfg['channel_range']}")

gen_id = 1_000 * rank

my_channel_range = channel_range.from_str(cfg["channel_range"][rank])
logger.info(f"rank: {rank} config: {cfg}")

#my_reader = reader_gen(cfg['shotnr'], gen_id, cfg["engine"], cfg["params"])
my_reader = reader_dataman(cfg)

logger.info("Instantiated reader")
my_reader.Open()

step = 0
logger.info("Waiting")

# while(True):
# #for i in range(10):
#     stepStatus = my_reader.BeginStep()
#     #print(stepStatus)
#     if stepStatus == adios2.StepStatus.OK:
#         var = dataman_IO.InquireVariable("floats")
#         shape = var.Shape()
#         io_array = np.zeros(np.prod(shape), dtype=np.float)
#         #reader.Get(var, io_array, adios2.Mode.Sync)
#         channel_data = my_reader.get_data("floats")
#         #currentStep = reader.CurrentStep()
#         my_reader.EndStep()
#         #print("rank {0:d}: Step".format(rank), reader.CurrentStep(), ", io_array = ", io_array)
#     else:
#         logger.info(f">>> receiver {rank:d}: End of stream")
#         break

#     # Recover channel data 
#     #channel_data = channel_data.reshape((num_channels, channel_data.size // num_channels))

#     logger.info(f">>> Step begins ... {step}")
#     ## jyc: this is just for testing. This is a place to run analysis if we want.
#     ##executor.submit(perform_analysis, channel_data, step)

#     ## Save data in a queue so that a workder thead will fetch and save concurrently.
#     #queue_list[step%num_analysis].put((channel_data, step))
#     step += 1


# End of file processor_simple.py