#!/usr/bin/env python
#-*- Encoding: UTF-8 -*-

"""
Do calculations with routines defined in kstar_ecei_helpers.py and verify against
results from the original KstarEcei object.
"""

import sys
import json

sys.path.append("/global/homes/r/rkube/repos/delta")
from analysis.channels import channel, channel_range
from analysis.ecei_helper import get_abcd, beam_path, channel_position

sys.path.append("/global/homes/r/rkube/repos/fluctana_rmc")
from kstarecei import KstarEcei

# Load a config file to get the ecei config
with open("../configs/test_skw.json", "r") as df: 
    cfg = json.load(df)
ecei_cfg = cfg["diagnostic"]["parameters"]
# Define a test channel
ch1 = channel('L', 17, 7)
print("Channel ", ch1)

# Calculate channel position using re-factored code
res = channel_position(ch1, ecei_cfg)

# Calculate channel positions using original code
K = KstarEcei(shot=18431, clist=["ECEI_" + ch1.__str__()], data_path='/global/cscratch1/sd/rkube/KSTAR/kstar_streaming/')

K.channel_position()
print("KStar:")
print("rpos = {0:6.4f}, zpos = {1:6.4f}, apos = {2:6.4f}".format(K.rpos[0], K.zpos[0], K.apos[0]))
print("Re-factored:")
print("rpos = {0:6.4f}, zpos = {1:6.4f}, apos = {2:6.4f}".format(*res))


""" Some results.
For Channel 1707 we have
rpos = 1.2975, zpos = 0.0815, apos = 0.0036

For Channel 0805 we have
rpos = 1.3243, zpos = -0.0814, apos = -0.0036

For Channel 1103 we have
rpos = 1.3522, zpos = -0.0271, apos = -0.0012

For Channel 0208 we have
rpos = 1.2845, zpos = -0.1902, apos = -0.0083


"""



# End of file test_kstar_ecei_helpers.py