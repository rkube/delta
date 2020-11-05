# -*- Encoding: UTF-8 -*-


# import json

# from ..analysis.kernels_spectral import kernel_crosscorr
# from ..analysis.kernels_spectral_cy import kernel_coherence_64_cy, kernel_crosspower_64_cy, kernel_crossphase_64_cy

# from ..data_models.channels_2d import channel_pair
# from ..data_models.helpers import gen_channel_range, unique_everseen

config_file = "../configs/test_all.json"

with open(config_file, "r") as df:
    cfg = json.load(df)


    
# print(cfg)



# End of file test_kernels_spectral.py