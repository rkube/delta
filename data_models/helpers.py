# -*- Encoding: UTF-8 -*-

"""Helper functions common to all data models"""


# def gen_channel_name(prefix: str, shotnr: int, channel_rg: str):
#     """Generates a channel name for KSTAR ECEI data"""
#     return f"{prefix}/{shotnr:05d}_ch{channel_rg:s}"

import numpy as np
import data_models.kstar_ecei

class data_model_generator():
    """Returns the data model for a given configuration"""
    def __init__(self, cfg_diagnostic: dict):
        """Sets up data model generation.

        Parameters:
        ===========
        cfg_diagnostic: dict,
                        Diagnostic section of the config file
        """
        self.cfg = cfg_diagnostic

        if self.cfg["name"] == "kstarecei":
            self.data_type = data_models.kstar_ecei.ecei_chunk
        elif self["name"] == "nstxgpi":
            self.data_type = None
        else:
            raise ValeuError(f"No data model for diagnotisc {cfg['diagnostic']['name']}")

    def new_chunk(self, stream_data: np.array, chunk_idx: int):
        """Generates a data model from new chunk of streamed data.

        Parameters
        ----------
        stream_data : np.array
                      New data chunk read from reader_gen.

        """

        # Generate a time-base and a data model
        if self.cfg["name"] == "kstarecei":
            # Adapt configuration file parameters for use in timebase_streaming
            # constructor
            t_start, t_end, _ = self.cfg["parameters"]["TriggerTime"]
            f_sample = 1e3 * self.cfg["parameters"]["SampleRate"]
            samples_per_chunk = self.cfg["datasource"]["chunk_size"]

            tb = data_models.kstar_ecei.timebase_streaming(t_start, t_end, f_sample,
                                               samples_per_chunk, chunk_idx)

            return data_models.kstar_ecei.ecei_chunk(stream_data, tb)

        elif self.cfg["name"] == "nstxgpi":
            raise NotImplementedError("NSTX chunk generation not implemented")

        else:
            raise NameError(f"Data model name not understood: {self.cfg['diagnostic']['name']}")




def gen_channel_name(cfg: dict) -> str:
    """Generates a name for the ADIOS channel from the diagnostic configuration"""

    if cfg["diagnostic"]["name"] == "kstarecei":
        experiment = "KSTAR"
        diagnostic = "ECEI"
        shotnr = int(cfg["diagnostic"]["shotnr"])
        channel_rg = cfg["diagnostic"]["datasource"]["channel_range"][0]

        channel_name = f"{experiment}_{shotnr:05d}_{diagnostic}_{channel_rg}"
        return channel_name

    else:
        raise ValueError


def gen_var_name(cfg: dict) -> str:
    """Generates a variable name from the diagnostic configuration"""

    if cfg["diagnostic"]["name"] == "kstarecei":
        return cfg["diagnostic"]["datasource"]["channel_range"]

    else:
        raise ValueError



# End of file helpers.py
