# -*- Encoding: UTF-8 -*-


"""
Author: Ralph Kube

Defines preprocessing functions
"""


def kernel_bandpass(data, params):
    """
    """
    pass



class pre_bandpass():
    """Implements bandpass filtering"""

    def __init__(self, params):
        self.params = params


    def process(self, data, executor):
        # Do nothing and return the data
        #result = executor.submit(kernel_bandpass, data, **self.params)
        return data
    






# End of file tasks.py