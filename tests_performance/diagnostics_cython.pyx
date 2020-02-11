# -*- Encoding: UTF-8 -*-
# ~/.conda/envs/intel/bin/cython diagnostics_cython.pyx

import numpy as np
cimport numpy as cnp
cimport cython
cimport openmp

from cython.parallel import prange

DTYPE = np.complex128
ctypedef cnp.complex128_t DTYPE_t


from libc.complex cimport conj, csqrt, cabs, creal


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def my_coherence_cy(cnp.ndarray[cnp.complex128_t, ndim=3] data, cnp.uint64_t[::1] ch1_idx_arr, cnp.uint64_t[::1] ch2_idx_arr):
    
    cdef size_t num_idx = ch1_idx_arr.size   # Length of index array
    cdef size_t num_fft = data.shape[1]   # Number of fft frequencies
    cdef size_t num_bins = data.shape[2]  # Number of ffts
    cdef size_t ch1_idx
    cdef size_t ch2_idx
    cdef size_t idx, nn, bb # Loop variables
    cdef double complex _tmp
    cdef double complex Pxx
    cdef double complex Pyy
    

    cdef cnp.ndarray[cnp.float64_t, ndim=2] result = np.zeros([num_idx, num_fft], dtype=np.float64)

    with nogil:
        for idx in prange(num_idx):
            ch1_idx = ch1_idx_arr[idx]
            ch2_idx = ch2_idx_arr[idx]

            for nn in range(num_fft):
                _tmp = 0.0
                for bb in range(num_bins):
                    Pxx = data[ch1_idx, nn, bb] * conj(data[ch1_idx, nn, bb])
                    Pyy = data[ch2_idx, nn, bb] * conj(data[ch2_idx, nn, bb])
                    _tmp = _tmp + data[ch1_idx, nn, bb] * conj(data[ch2_idx, nn, bb]) / csqrt(Pxx * Pyy)

                #_tmp /= num_bins

                result[idx, nn] = creal(cabs(_tmp)) / num_bins

    return(result)

# End of file diagnostics_cython.pyx