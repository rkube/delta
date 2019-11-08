#Coding: UTF-8 -*-

from analysis.spectral import power_spectrum


def analyze_and_store(channel_data, my_analysis, backend):
    """Analyze and store data

    channel_data:  ndarray, float: data to be analyzed
    method: string, name of the analysis method
    backend: obj, callable: storage backend

    """

    #print("Analyze and store")

    if my_analysis["name"] == "power_spectrum":
        result = power_spectrum(channel_data, **my_analysis["config"])

    backend.store(my_analysis, result)


# End of file analyze.py