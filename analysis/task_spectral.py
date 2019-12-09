# coding: UTF-8 -*-

from analysis.channels import channel, channel_range
import itertools


class task_spectral():
    """Serves as the super-class for analysis methods. Do not instantiate directly"""

    def __init__(self, task_config, fft_config):
        """Initialize the object with a fixed channel list, a fixed name of the analysis to be performed
        and a fixed set of parameters for the analysis routine.

        Inputs:
        =======
        channel_range: list of strings, defines the name of the channels. This should probably match the
                      name of the channels in the BP file.
        task_config: dict, defines parameters of the analysis to be performed
        fft_config dict, gives parameters of the fourier-transformed data
        """


        # Stores the description of the task. This can be arbitrary
        self.description = task_config["description"]
        # Stores the name of the analysis we are going to execute
        self.analysis = task_config["analysis"]
        
        # Parse the reference and cross channels.
        try:
            kwargs = task_config["kwargs"]
            # These channels serve as reference for the spectral diagnostics
            self.ref_channels = channel_range.from_str(kwargs["ref_channels"][0])
            # These channels serve as the cross-data for the spectrail diagnostics
            self.x_channels = channel_range.from_str(kwargs["x_channels"][0])
        except KeyError:
            self.kwargs = None

        self.fft_config = fft_config

        self.storage_scheme =  {"ref_channels": self.ref_channels.to_str(),
                                "cross_channels": self.x_channels.to_str()}


    def calculate(self, *args):
        raise NotImplementedError


    def get_dispatch_sequence(self):
        """Returns an iterator over the reference and cross channels."""

        # Chain the iterators over ref_channels and x_channels into one iterator
        crg_total = itertools.chain(self.ref_channels, self.x_channels)
        # Define and return a new iterator over combinations with replacements of these two
        # iterators
        return(itertools.combinations_with_replacement(crg_total, 2))


class task_cross_phase(task_spectral):
    """This class calculates the cross-phase via the calculate method."""
    def __init__(self, task_config, fft_config):
        super().__init__(task_config, fft_config)
        # Append the analysis name to the storage scheme
        self.storage_scheme["analysis_name"] = "cross_phase"



    def calculate(self, dask_client, fft_future):
        """Calculates the cross phase of signal data.
        The data is assumed to be distributed to the clients.
        Before calling this method the following steps needs to be done:
        
        Set up a task
        >>> task = task_cross_phase(task_config, fft_config)
        Scatter time-chunk data to the cluster
        >>> data_future = client.scatter(data, broadcast=True)
        Calculate the fft. Do this with a special method, so we don't use dask array
        >>> fft_future = my_fft.do_fft(client, data_future)
        Gather the results
        >>> results = client.gather(fft_future)
        Create an np array from the fourier-transformed data
        >>> fft_data = np.array(results)
        Scatter the dask array to all clients.
        >>> fft_future = client.scatter(fft_data, broadcast=True)
        Execute a task on the transformed data
        >>> task.calculate(dask_client, fft_future)        
        


        Input:
        ======
        client: dask client
        fft_future: A future to the fourier-transformed data. 

        Output:
        =======
        future: dask future that holds the result of the analysis.
        """

        # Somehow dask complains when we don't define the function in the local scope.
        def cross_phase(fft_data, ch0, ch1):
            """Kernel that calculates the cross-phase between two channels.
            Input:
            ======
            fft_data: ndarray, float: Contains the fourier-transformed data. 
                      dim0: channel, dim1: Fourier Coefficients, dim2: STFT (bins in fluctana code)
            ch0: int, index for first channel
            ch1: int, index for second channel

            Returns:
            ========
            cp: float, the cross phase
            """    

            from math import atan2
            _tmp1 = (fft_data[ch0, :, :] * fft_data[ch1, :, :].conj()).mean(axis=1)#.compute()
            return(atan2(_tmp1.real, _tmp1.imag).real)


        self.futures_list = [dask_client.submit(cross_phase, fft_future, ch_r.idx(), ch_x.idx()) for ch_r, ch_x in self.get_dispatch_sequence()]

        return None


class task_cross_power(task_spectral):
    """This class calculates the cross-power between two channels."""
    def __init__(self, task_config, fft_config):
        super().__init__(task_config, fft_config)
        self.storage_scheme["analysis_name"] = "cross_power"

    def calculate(self, dask_client, fft_future):
        def cross_power(fft_data, ch0, ch1):
            """Kernel that calculates the cross-power between two channels.
            Input:
            ======    
            fft_data: ndarray, float: Contains the fourier-transformed data. 
                      dim0: channel, dim1: Fourier Coefficients, dim2: STFT (bins in fluctana code)
            ch0: int, index for first channel
            ch1: int, index for second channel

            Returns:
            ========
            cross_power, float.
            """
            return((fft_data[ch0, :] * fft_data[ch1, :].conj()).mean().__abs__())
    

        self.futures_list = [dask_client.submit(cross_power, fft_future, ch_r.idx(), ch_x.idx()) for ch_r, ch_x in self.get_dispatch_sequence()]
        return None        


class task_coherence(task_spectral):
    """This class calculates the coherence between two channels."""
    def __init__(self, task_config, fft_config):
        super().__init__(task_config, fft_config)
        self.storage_scheme["analysis_name"] = "coherence"

    
    def calculate(self, dask_client, fft_future):
        def coherence(fft_data, ch0, ch1):
            """Kernel that calculates the coherence between two channels.
            Input:
            ======    
            fft_data: ndarray, float: Contains the fourier-transformed data. 
                      dim0: channel, dim1: Fourier Coefficients. dim2: STFT (bins in fluctana code)
            ch0: int, index for first channel
            ch1: int, index for second channel

            Returns:
            ========
            coherence, float.
            """

            from numpy import sqrt, mean, fabs
            X = fft_data[ch0, :, :]
            Y = fft_data[ch1, :, :]

            Pxx = X * X.conj()
            Pyy = Y * Y.conj()

            Gxy = mean((X * Y.conj()) / sqrt(Pxx * Pyy), axis=1)
            Gxy = fabs(Gxy).real

            #Gxy = fabs(mean(X * Y.conj() / sqrt(X * X.conj() * Y * Y.conj())).real)

            return(Gxy)

        self.futures_list = [dask_client.submit(coherence, fft_future, ch_r.idx(), ch_x.idx()) for ch_r, ch_x in self.get_dispatch_sequence()]
        return None  

#    def store(self, mongo_client):
#        for future in future_list:
#            dask_client.submit(mongo_client.store, future)


#class mongo_client:
#    def store(self, future):
#        self.collection.insert_one(future.result())


class task_xspec(task_spectral):
    """This class calculates the coherence between two channels."""
    def __init__(self, task_config, fft_config):
        super().__init__(task_config, fft_config)
        self.storage_scheme["analysis_name"] = "xspec"
    
    def calculate(self, dask_client, fft_future):
        def xspec(fft_data, ch0, ch1):
            """Kernel that calculates the coherence between two channels.
            Input:
            ======    
            fft_data: dask_array, float: Contains the fourier-transformed data. 
                      dim0: channel, dim1: Fourier Coefficients, dim2: STFT (bins in fluctana code)
            ch0: int, index for first channel
            ch1: int, index for second channel

            Returns:
            ========
            coherence, float.
            """

            return 0.0

        #self.futures_list = [dask_client.submit(coherence, fft_future, ch_r.idx(), ch_x.idx()) for ch_r in self.ref_channels for ch_x in self.x_channels]
        raise NotImplementedError
        return None  


class task_cross_correlation(task_spectral):
    """This class calculates the cross-correlation between two channels."""
    def __init__(self, task_config, fft_config):
        super().__init__(task_config, fft_config)
        self.storage_scheme["analysis_name"] = "cross_correlation"

    def calculate(self, dask_client, fft_future):
        def cross_corr(fft_data, ch0, ch1, fft_params):
            """Defines a kernel that calculates the cross-correlation between two channels.

            Input:
            ======
            fft_data: ndarray, float: Contains the fourier-transformed data. 
                      dim0: channel. dim1: Fourier Coefficients, dim2: STFT (bins in fluctana code)
            ch0: int, index of first channel
            ch1: int, index of second channel
            fft_params: dict, parameters of the fourier-transformed data

            Returns:
            ========
            cross-correlation, float array
            """

            from numpy.fft import ifftshift, ifft, fftshift

            # Perform fftshift on the fourier coefficient axis (dim1)
            X = fftshift(fft_data[ch0, :, :], axes=0)
            Y = fftshift(fft_data[ch1, :, :], axes=0)

            _tmp = ifftshift(X * Y.conj(), axes=0) / fft_params['win_factor']
            _tmp = ifft(_tmp, n=fft_params['nfft'], axis=0) * fft_params['nfft']
            _tmp = fftshift(_tmp, axes=0)

            res = _tmp.mean(axis=1).real

            return(res)


        #print("task_cross_corr: fft_config = ", self.fft_config)
        #print("Dispatch sequence")
        #for ch_r, ch_x in self.get_dispatch_sequence():
        #    print(ch_r, ch_x)

        self.futures_list = [dask_client.submit(cross_corr, fft_future, ch_r.idx(), ch_x.idx(), self.fft_config) for ch_r, ch_x in self.get_dispatch_sequence()]
        return None 


class task_bicoherence(task_spectral):
    """This class calculates the bicoherence between two channels."""
    def __init__(self, task_config, fft_config):
        super().__init__(task_config, fft_config)
        self.storage_scheme["analysis_name"] = "xspec"
    
    def calculate(self, dask_client, fft_future):
        def bicoherence(fft_data, ch0, ch1): 
            """Kernel that calculates the bi-coherence between two channels.
            Input:
            ======    
            fft_data: dask_array, float: Contains the fourier-transformed data. dim0: channel, dim1: Fourier Coefficients
            ch0: int, index for first channel
            ch1: int, index for second channel

            Returns:
            ========
            bicoherence, float.
            """
            import numpy as np

            # Transpose to make array layout compatible with code from specs.py
            XX = np.fft.fftshift(fft_data[ch0, :, :], axes=0).T
            YY = np.fft.fftshift(fft_data[ch1, :, :], axes=0).T

            bins, full = XX.shape
            half = full // 2 + 1

            # calculate bicoherence
            B = np.zeros((full, half), dtype=np.complex_)
            P12 = np.zeros((full, half))
            P3 = np.zeros((full, half))
            val = np.zeros((full, half))

            for b in range(bins):
                X = XX[b,:] # full -fN ~ fN
                Y = YY[b,:] # full -fN ~ fN

                Xhalf = np.fft.ifftshift(X) # full 0 ~ fN, -fN ~ -f1
                Xhalf = Xhalf[0:half] # half 0 ~ fN

                X1 = np.transpose(np.tile(X, (half, 1)))
                X2 = np.tile(Xhalf, (full, 1))
                X3 = np.zeros((full, half), dtype=np.complex_)
                for j in range(half):
                    if j == 0:
                        X3[0:, j] = Y[j:]
                    else:
                        X3[0:(-j), j] = Y[j:]

                B = B + X1 * X2 * np.matrix.conjugate(X3) / bins #  complex bin average
                P12 = P12 + (np.abs(X1 * X2).real)**2 / bins # real average
                P3 = P3 + (np.abs(X3).real)**2 / bins # real average

            # val = np.log10(np.abs(B)**2) # bispectrum
            val = (np.abs(B)**2) / P12 / P3 # bicoherence

            # summation over pairs
            sum_val = np.zeros(full)
            for i in range(half):
                if i == 0:
                    sum_val = sum_val + val[:,i]
                else:
                    sum_val[i:] = sum_val[i:] + val[:-i,i]

            N = np.array([i+1 for i in range(half)] + [half for i in range(full-half)])
            sum_val = sum_val / N # element wise division

            return (val, sum_val)

        self.futures_list = [dask_client.submit(bicoherence, fft_future, ch_r.idx(), ch_x.idx()) for ch_r, ch_x in self.get_dispatch_sequence()]
        return None 





    #    # 1)
    #     if self.analysis == "cwt":
    #         raise NotImplementedError
    #     # 3)
    #     elif self.analysis == "coherence":
    #         raise NotImplementedError        


    #     # 6)
    #     elif self.analysis == "corr_coeff":
    #         raise NotImplementedError


    #     # 8)
    #     elif self.analysis == "skw":
    #         raise NotImplementedError


    #     print(self.futures_list)


# End of file analysis_package.py