#-*- Encoding: UTF-8 -*-

"""
Load 3 time chunks of data and write store it as numpy files.
This data is used as verification for the diagnostics kernels
and benchmarked against the fluctana routines
"""

import json
from readers.reader_one_to_one import reader_bpfile


def main():

    # Open the simple reader config
    with open("tests_analysis/config_generate_test_data.json", "r") as df:
        cfg = json.load(df)
        df.close()

    reader = reader_bpfile(cfg["shotnr"], cfg["diagnostic"]["parameters"])
    reader.Open(cfg["datapath"])

    while True:
        stepStatus = reader.BeginStep()

        if stepStatus:
            # Read data
            stream_data = reader.Get(save=True)

        if reader.CurrentStep() >= 3:
            break


if __name__ == "__main__":
    main()