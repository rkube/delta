# coding: UTF-8 -*-

import numpy as np
from analysis.channels import channel, channel_range, channel_pair, unique_everseen
import more_itertools

"""
Author: Ralph Kube
This file contains the task_spectral class and its derived classes. Each one implements
an analysis routine from the fluctana package.

The parent class, task_spectral, and implements common methods to all tasks:
* It handles the range of channels for which any analysis will be performed
* It defines how this channel range is divided into sub-ranges
* It defines a storage scheme for results of an analysis
* It defines a storage scheme for the meta data

Each child class defines a unique analysis routine in the calculate method
that is applied to a data chunk. For this it uses a poolexecuter model.
The executor client is called with the tasks analysis method, the data chunk
and a channel range. The results of this calculation are accessible through its
future_list.
"""



class task_spectral():
    """Serves as the super-class for analysis methods. Do not instantiate directly"""

    def __init__(self, task_config, fft_config, ecei_config):
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
        self.ref_channels = channel_range.from_str(task_config["ref_channels"])
        # These channels serve as the cross-data for the spectral diagnostics
        self.cmp_channels = channel_range.from_str(task_config["cmp_channels"])


        self.task_config = task_config
        self.fft_config = fft_config
        self.ecei_config = ecei_config

        self.storage_scheme =  {"ref_channels": self.ref_channels.to_str(),
                                "cmp_channels": self.cmp_channels.to_str()}

        # Construct a list of unique channels
        # F.ex. we have ref_channels [(1,1), (1,2), (1,3)] and cmp_channels = [(1,1), (1,2)]
        # The unique list of channels is then
        # (1,1) x (1,1), (1,1) x (1,2)
        # (1,2) x (1,2) !!! Omit (1,2) x (1,1)
        # (1,3) x (1,1)
        # (1,3) x (1,2)
        channel_pairs = [channel_pair(cr, cx) for cr in self.ref_channels for cx in self.cmp_channels]
        # Make a list, so that we don't exhause the iterator after the first call.
        self.unique_channels = list(more_itertools.distinct_combinations(channel_pairs, 1))
        self.channel_chunk_size = task_config["channel_chunk_size"]


    def calculate(self, *args):
        raise NotImplementedError


    def get_dispatch_sequence(self, niter=None):
        """Returns an a list of iterables that together span all unique
        combinations of ref_ch x cmp_ch.

        Parameters:
        ===========
        niter, int: Length of the sub-lists we split the list of channel pairs into.
        """

        if niter is None:
            niter = self.channel_chunk_size

        all_chunks = more_itertools.chunked(self.unique_channels, niter)
        return(all_chunks)


    def store_data(self, backend, metadata):
        """Store results of computation in the backend.

        Input:
        ======
        backend: Backend to use
        metadata, dict: Dictionary that is passed into the backend call
        """

        # Gather the results
        print("*** Storing data. len(futures_list) = {0:d}".format(len(self.futures_list)))
        res = []
        for f in self.futures_list:
            res.append(f.result())
        res = np.concatenate(res, axis=0)

        #print("*** Done. futures_list = ", self.futures_list)
        print("*** res.shape = ", res.shape)
        backend.store(self.description, res, metadata)
        
        
    def store_metadata(self, backend):
        """Store meta-data that only depends on task configuration.
        
        Input:
        ======
        backend, Backend type to use
        """

        # Get the channel list
        all_chunks = self.get_dispatch_sequence()

        ll = list(all_chunks)
        flat_ll = [item for sublist in ll for item in sublist]
        flat_ll = [l[0] for l in flat_ll]


        metadata = {"task_config": self.task_config, 
                    "fft_config": self.fft_config,
                    "ecei_config": self.ecei_config,
                    "channel_list": flat_ll}

        backend.store_config(self.description, metadata)

        


class task_cross_phase(task_spectral):
    """This class calculates the cross-phase via the calculate method."""
    def __init__(self, task_config, fft_config, ecei_config):
        super().__init__(task_config, fft_config, ecei_config)
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
        def cross_phase(fft_data, ch_it):
            """Kernel that calculates the cross-phase between two channels.
            Input:
            ======
            fft_data: ndarray, float: Contains the fourier-transformed data. 
                      dim0: channel, dim1: Fourier Coefficients, dim2: STFT (bins in fluctana code)
            ch_it: iterable, Iterator over a list of channels we wish to perform our computation on

            Returns:
            ========
            Axy: float, the cross phase
            """    
            import numpy as np
            
            c1_idx = np.array([cc[0].ch1.idx() for cc in ch_it])
            c2_idx = np.array([cc[0].ch2.idx() for cc in ch_it])
            Pxy = (fft_data[c1_idx, :, :] * fft_data[c2_idx, :, :].conj()).mean(axis=2)
            return(np.arctan2(Pxy.real, Pxy.imag).real)

        self.futures_list = [dask_client.submit(cross_phase, fft_future, ch_it) for ch_it in self.get_dispatch_sequence()]

        return None


class task_cross_power(task_spectral):
    """This class calculates the cross-power between two channels."""
    def __init__(self, task_config, fft_config, ecei_config):
        super().__init__(task_config, fft_config, ecei_config)
        self.storage_scheme["analysis_name"] = "cross_power"

    def calculate(self, dask_client, fft_future):
        def cross_power(fft_data, ch_it, fft_config):
            """Kernel that calculates the cross-power between two channels.
            Input:
            ======    
            fft_data: ndarray, float: Contains the fourier-transformed data. 
                      dim0: channel, dim1: Fourier Coefficients, dim2: STFT (bins in fluctana code)
            ch_it: iterable, Iterator over a list of channels we wish to perform our computation on


            Returns:
            ========
            cross_power, float.
            """
            import numpy as np
            
            c1_idx = np.array([cc[0].ch1.idx() for cc in ch_it])
            c2_idx = np.array([cc[0].ch2.idx() for cc in ch_it])

            res = (fft_data[c1_idx, :, :] * fft_data[c2_idx, :, :].conj()).mean(axis=2) / fft_config["win_factor"]
            
            return(res)
    

        self.futures_list = [dask_client.submit(cross_power, fft_future, ch_it, self.fft_config) for ch_it in self.get_dispatch_sequence()]
        return None        


class task_coherence(task_spectral):
    """This class calculates the coherence between two channels."""
    def __init__(self, task_config, fft_config, ecei_config):
        super().__init__(task_config, fft_config, ecei_config)
        self.storage_scheme["analysis_name"] = "coherence"

    
    def calculate(self, dask_client, fft_future):
        def coherence(fft_data, ch_it):
            """Kernel that calculates the coherence between two channels.
            Input:
            ======    
            fft_data: ndarray, float: Contains the fourier-transformed data. 
                      dim0: channel, dim1: Fourier Coefficients. dim2: STFT (bins in fluctana code)
            ch_it: iterable, Iterator over a list of channels we wish to perform our computation on


            Returns:
            ========
            coherence, float.
            """

            import numpy as np

            c1_idx = np.array([cc[0].ch1.idx() for cc in ch_it])
            c2_idx = np.array([cc[0].ch2.idx() for cc in ch_it])

            X = fft_data[c1_idx, :, :]
            Y = fft_data[c2_idx, :, :]

            Pxx = X * X.conj()
            Pyy = Y * Y.conj()

            Gxy = np.mean((X * Y.conj()) / np.sqrt(Pxx * Pyy), axis=2)
            Gxy = np.fabs(Gxy).real

            #Gxy = fabs(mean(X * Y.conj() / sqrt(X * X.conj() * Y * Y.conj())).real)

            return(Gxy)

        self.futures_list = [dask_client.submit(coherence, fft_future, ch_it) for ch_it in self.get_dispatch_sequence()]
        return None  

#    def store(self, mongo_client):
#        for future in future_list:
#            dask_client.submit(mongo_client.store, future)


#class mongo_client:
#    def store(self, future):
#        self.collection.insert_one(future.result())


class task_xspec(task_spectral):
    """This class calculates the coherence between two channels."""
    def __init__(self, task_config, fft_config, ecei_config):
        super().__init__(task_config, fft_config, ecei_config)
        self.storage_scheme["analysis_name"] = "xspec"
    
    def calculate(self, dask_client, fft_future):
        def xspec(fft_data, ch_it):
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

        raise NotImplementedError


class task_cross_correlation(task_spectral):
    """This class calculates the cross-correlation between two channels."""
    def __init__(self, task_config, fft_config, ecei_config):
        super().__init__(task_config, fft_config, ecei_config)
        self.storage_scheme["analysis_name"] = "cross_correlation"

    def calculate(self, dask_client, fft_future):
        def cross_corr(fft_data, ch_it, fft_params):
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

            #import numpy as np


            c1_idx = np.array([cc[0].ch1.idx() for cc in ch_it])
            c2_idx = np.array([cc[0].ch2.idx() for cc in ch_it])

            # Perform fftshift on the fourier coefficient axis (dim1)
            X = np.fft.fftshift(fft_data[c1_idx, :, :], axes=1)
            Y = np.fft.fftshift(fft_data[c2_idx, :, :], axes=1)

            _tmp = np.fft.ifftshift(X * Y.conj(), axes=1) / fft_params['win_factor']
            _tmp = np.fft.ifft(_tmp, n=fft_params['nfft'], axis=1) * fft_params['nfft']
            _tmp = np.fft.fftshift(_tmp, axes=1)

            res = _tmp.mean(axis=2).real

            return(res)


        #print("task_cross_corr: fft_config = ", self.fft_config)
        #print("Dispatch sequence")
        #for ch_r, ch_x in self.get_dispatch_sequence():
        #    print(ch_r, ch_x)

        self.futures_list = [dask_client.submit(cross_corr, fft_future, ch_it, self.fft_config) for ch_it in self.get_dispatch_sequence()]
        return None 


class task_bicoherence(task_spectral):
    """This class calculates the bicoherence between two channels."""
    def __init__(self, task_config, fft_config, ecei_config):
        super().__init__(task_config, fft_config, ecei_config)
        self.storage_scheme["analysis_name"] = "xspec"
    
    def calculate(self, dask_client, fft_future):
        def bicoherence(fft_data, ch_it): 
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

            res_list = []

            for ch in ch_it:
                ch1_idx, ch2_idx = ch[0].ch1.idx(), ch[0].ch2.idx()

                # Transpose to make array layout compatible with code from specs.py
                XX = np.fft.fftshift(fft_data[ch1_idx, :, :], axes=0).T
                YY = np.fft.fftshift(fft_data[ch2_idx :, :], axes=0).T

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

                res_list.append(val, sum_val)

            return (res_list)

        self.futures_list = [dask_client.submit(bicoherence, fft_future, ch_it) for ch_it in self.get_dispatch_sequence()]
        return None 

class task_skw(task_spectral):
    """This class calculates the bicoherence between two channels."""
    def __init__(self, task_config, fft_config, ecei_config):
        super().__init__(task_config, fft_config, ecei_config)
        self.storage_scheme["analysis_name"] = "xspec"
    
    def calculate(self, dask_client, fft_future):
        def skw(fft_data, ch_it, fft_params, ecei_config, kstep=0.01): 
            """
            Calculates the conditional spectrum S(k,w).

            Input:
            ======
            fft_data: dask_array, float: Contains the fourier-transformed data. dim0: channel, dim1: Fourier Coefficients
            ch0: channel, first channel
            ch1: channel, second channel
            fft_params: dictionary, parameters for fft
            ecei_config: dictionary, configuration of ecei diagnostic


            Returns:
            ========
            bicoherence, float.
            """

            import numpy as np
            from analysis.ecei_helper import channel_position

            res_list = []
            for ch in ch_it:
                ch1_idx, ch2_idx = ch[0].ch1.idx(), ch[0].ch2.idx()
                print("Calculating skw for channels {0:s}x{1:s}".format(ch[0].ch1, ch[0].ch2))

                XX = np.fft.fftshift(fft_data[ch1_idx, :, :], axes=0).T
                YY = np.fft.fftshift(fft_data[ch2_idx, :, :], axes=0).T

                bins, _ = XX.shape
                win_factor = fft_params["win_factor"]

                cpos_ref = channel_position(ch1_idx, ecei_config)
                cpos_cmp = channel_position(ch2_idx, ecei_config)

                # Calculate distance between channels
                dist = np.sqrt( (cpos_ref[0] - cpos_cmp[0])**2.0 + (cpos_ref[1] - cpos_cmp[1])**2.0)
                dmin = dist * 1e2

                kax = np.arange(-np.pi / dmin, np.pi / dmin, kstep)

                nkax = kax.size
                nfft = fft_params["nfft"]

                if(ch1_idx == ch2_idx):
                    # We can't calculate the cross-conditional spectrum for ch0==ch1
                    # since dmin=0
                    return(np.zeros(nkax, nfft))

                # value dimension
                Pxx = np.zeros((bins, nfft), dtype=np.complex_)
                Pyy = np.zeros((bins, nfft), dtype=np.complex_)
                Kxy = np.zeros((bins, nfft), dtype=np.complex_)
                val = np.zeros((nkax, nfft), dtype=np.complex_)

                sklw = np.zeros((nkax, nfft), dtype=np.complex_)
                K = np.zeros(nfft, dtype=np.complex_)
                sigK = np.zeros(nfft, dtype=np.complex_)


                print("nkax = {0:d}, nfft = {1:d}, dmin = {2:f}".format(nkax, nfft, dmin))

                # calculate auto power and cross phase (wavenumber)
                for b in range(bins):
                    #X = self.Dlist[done].spdata[done_subset[c],b,:]
                    #Y = self.Dlist[dtwo].spdata[dtwo_subset[c],b,:]
                    X = XX[b, :]
                    Y = YY[b, :]

                    Pxx[b,:] = X*np.matrix.conjugate(X) / win_factor
                    Pyy[b,:] = Y*np.matrix.conjugate(Y) / win_factor
                    Pxy = X*np.matrix.conjugate(Y)
                    Kxy[b,:] = np.arctan2(Pxy.imag, Pxy.real).real / (dist * 100) # [cm^-1]
                                                                    
                    # calculate SKw
                    for w in range(nfft):
                        idx = (Kxy[b,w] - kstep * 0.5 < kax) * (kax < Kxy[b,w] + kstep * 0.5)
                        val[:,w] = val[:,w] + (1.0 / bins * (Pxx[b,w] + Pyy[b,w]) * 0.5) * idx

                # calculate moments
                sklw = val / np.tile(val.sum(axis=0), (nkax, 1))
                K[:] = np.sum(np.transpose(np.tile(kax, (nfft, 1))) * sklw, axis=0)
                for w in range(nfft):
                    sigK[w] = np.sqrt(np.sum( (kax - K[w])**2 * sklw[:,w] ))

                val = val.mean(axis=0).real
                K = np.mean(K, axis=0)
                sigK = np.mean(sigK, axis=0)

                pdata = np.log10(val + 1e-10)

                print(pdata.shape)

                res_list.append(pdata)

            return(res_list)


        for c in self.get_dispatch_sequence():
            print(c.ch1.__str__(), c.ch2.__str__())

        self.futures_list = [dask_client.submit(skw, fft_future, ch_it, self.fft_config, self.ecei_config) for ch_it in self.get_dispatch_sequence()]
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



    #     print(self.futures_list)


# End of file analysis_package.py