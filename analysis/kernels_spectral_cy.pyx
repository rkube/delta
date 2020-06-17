# -*- Encoding: UTF-8 -*-

import numpy as np
cimport numpy as cnp
cimport cython
cimport openmp

from cython.parallel import prange

#from libc.complex cimport conj, csqrt, cabs, creal, cimag
cimport cython
cdef extern from "complex.h" nogil:
    double complex conj(double complex x)
    double complex csqrt(double complex x)
    double cabs(double complex x)
    double creal(double complex x)
    double cimag(double complex x)

from libc.math cimport atan2
from libc.stdint cimport uint32_t

#cdef extern from "kernels.h":
#    void kernel_coherence_64(double complex* fft_data,
#                             double* result,
#                             size_t* ch1_idx_arr,
#                             size_t* ch2_idx_arr,
#                             size_t num_idx,
#                             size_t num_fft,
#                             size_t num_bins)
#
#
#    void kernel_coherence_32(float complex* fft_data,
#                             float* result,
#                             size_t* ch1_idx_arr,
#                             size_t* ch2_idx_arr,
#                             size_t num_idx,
#                             size_t num_fft,
#                             size_t num_bins)
#
#
#    void kernel_crossphase_64(double complex* fft_data,
#                              double* result,
#                              size_t* ch1_idx_arr,
#                              size_t* ch2_idx_arr,
#                              size_t num_idx,
#                              size_t num_fft,
#                              size_t num_bins)
#
#
#    void kernel_crossphase_32(float complex* fft_data,
#                              float* result,
#                              size_t* ch1_idx_arr,
#                              size_t* ch2_idx_arr,
#                              size_t num_idx,
#                              size_t num_fft,
#                              size_t num_bins)
#
#
#    void kernel_crosspower_64(double complex* fft_data,
#                              double* result,
#                              size_t* ch1_idx_arr,
#                              size_t* ch2_idx_arr,
#                              size_t num_idx,
#                              size_t num_fft,
#                              size_t num_bins)
#
#
#    void kernel_crosspower_32(float complex* fft_data,
#                              float* result,
#                              size_t* ch1_idx_arr,
#                              size_t* ch2_idx_arr,
#                              size_t num_idx,
#                              size_t num_fft,
#                              size_t num_bins)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def kernel_coherence_64_cy(cnp.ndarray[cnp.complex128_t, ndim=3] data, ch_it, fft_config):
    cdef size_t num_idx = len(ch_it)      # Length of index array
    cdef size_t num_fft = data.shape[1]   # Number of fft frequencies
    cdef size_t num_bins = data.shape[2]  # Number of ffts
    cdef size_t ch1_idx
    cdef size_t ch2_idx
    cdef size_t idx, nn, bb # Loop variables
    cdef double complex _tmp
    cdef double complex Pxx
    cdef double complex Pyy
    
    cdef cnp.ndarray[cnp.uint64_t, ndim=1] ch1_idx_arr = np.array([int(ch_pair.ch1.idx()) for ch_pair in ch_it], dtype=np.uint64)
    cdef cnp.ndarray[cnp.uint64_t, ndim=1] ch2_idx_arr = np.array([int(ch_pair.ch2.idx()) for ch_pair in ch_it], dtype=np.uint64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] result = np.zeros([num_idx, num_fft], dtype=np.float64)
 
    with nogil: 
        num_threads = openmp.omp_get_num_threads()
        for idx in prange(num_idx, schedule=static):
            ch1_idx = ch1_idx_arr[idx]
            ch2_idx = ch2_idx_arr[idx]

            for nn in range(num_fft):
                _tmp = 0.0
                for bb in range(num_bins):
                    Pxx = data[ch1_idx, nn, bb] * conj(data[ch1_idx, nn, bb])
                    Pyy = data[ch2_idx, nn, bb] * conj(data[ch2_idx, nn, bb])
                    _tmp = _tmp + data[ch1_idx, nn, bb] * conj(data[ch2_idx, nn, bb]) / csqrt(Pxx * Pyy)

                result[idx, nn] = creal(cabs(_tmp)) / num_bins

    return(result) 


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def kernel_coherence_32_cy(cnp.ndarray[cnp.complex64_t, ndim=3] data, ch_it, fft_config):
    cdef size_t num_idx = len(ch_it)      # Length of index array
    cdef size_t num_fft = data.shape[1]   # Number of fft frequencies
    cdef size_t num_bins = data.shape[2]  # Number of ffts
    cdef size_t ch1_idx
    cdef size_t ch2_idx
    cdef size_t idx, nn, bb # Loop variables
    cdef double complex _tmp
    cdef double complex Pxx
    cdef double complex Pyy
    
    cdef cnp.ndarray[cnp.uint64_t, ndim=1] ch1_idx_arr = np.array([int(ch_pair.ch1.idx()) for ch_pair in ch_it], dtype=np.uint64)
    cdef cnp.ndarray[cnp.uint64_t, ndim=1] ch2_idx_arr = np.array([int(ch_pair.ch2.idx()) for ch_pair in ch_it], dtype=np.uint64)
    cdef cnp.ndarray[cnp.float32_t, ndim=2] result = np.zeros([num_idx, num_fft], dtype=np.float32)
 
    with nogil: 
        for idx in prange(num_idx, schedule=static):
            ch1_idx = ch1_idx_arr[idx]
            ch2_idx = ch2_idx_arr[idx]

            for nn in range(num_fft):
                _tmp = 0.0
                for bb in range(num_bins):
                    Pxx = data[ch1_idx, nn, bb] * conj(data[ch1_idx, nn, bb])
                    Pyy = data[ch2_idx, nn, bb] * conj(data[ch2_idx, nn, bb])
                    _tmp = _tmp + data[ch1_idx, nn, bb] * conj(data[ch2_idx, nn, bb]) / csqrt(Pxx * Pyy)

                result[idx, nn] = creal(cabs(_tmp)) / num_bins

    return(result) 


#def kernel_coherence_64_v2(cnp.ndarray[cnp.complex128_t, ndim=3] data, ch_it, fft_config):
#    cdef size_t num_idx = len(ch_it)      # Length of index array
#    cdef size_t num_fft = data.shape[1]   # Number of fft frequencies
#    cdef size_t num_bins = data.shape[2]  # Number of ffts
#    cdef cnp.ndarray[cnp.uintp_t, ndim=1] ch1_idx_arr = np.array([np.uintp(ch_pair.ch1.idx()) for ch_pair in ch_it], dtype=np.uint64)
#    cdef cnp.ndarray[cnp.uintp_t, ndim=1] ch2_idx_arr = np.array([np.uintp(ch_pair.ch2.idx()) for ch_pair in ch_it], dtype=np.uint64)
#    cdef cnp.ndarray[cnp.float64_t, ndim=2] result = np.zeros([num_idx, num_fft], dtype=np.float64)
#
#    kernel_coherence_64(&data[0, 0, 0], &result[0, 0], &ch1_idx_arr[0], &ch2_idx_arr[0], num_idx, num_fft, num_bins)
# 
#    return(result)
# 
# 
#def kernel_coherence_32_v2(cnp.ndarray[cnp.complex64_t, ndim=3] data, ch_it, fft_config):
#    cdef size_t num_idx = len(ch_it)      # Length of index array
#    cdef size_t num_fft = data.shape[1]   # Number of fft frequencies
#    cdef size_t num_bins = data.shape[2]  # Number of ffts
#    cdef cnp.ndarray[cnp.uintp_t, ndim=1] ch1_idx_arr = np.array([np.uintp(ch_pair.ch1.idx()) for ch_pair in ch_it], dtype=np.uint64)
#    cdef cnp.ndarray[cnp.uintp_t, ndim=1] ch2_idx_arr = np.array([np.uintp(ch_pair.ch2.idx()) for ch_pair in ch_it], dtype=np.uint64)
#    cdef cnp.ndarray[cnp.float32_t, ndim=2] result = np.zeros([num_idx, num_fft], dtype=np.float32)
#
#    kernel_coherence_32(&data[0, 0, 0], &result[0, 0], &ch1_idx_arr[0], &ch2_idx_arr[0], num_idx, num_fft, num_bins)
# 
#    return(result)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def kernel_crossphase_64_cy(cnp.ndarray[cnp.complex128_t, ndim=3] data, ch_it, fft_config):
    
    cdef size_t num_idx = len(ch_it)      # Length of index array
    cdef size_t num_fft = data.shape[1]   # Number of fft frequencies
    cdef size_t num_bins = data.shape[2]  # Number of ffts
    cdef size_t ch1_idx
    cdef size_t ch2_idx
    cdef size_t idx, nn, bb # Loop variables
    cdef double complex _tmp

    cdef cnp.ndarray[cnp.uint64_t, ndim=1] ch1_idx_arr = np.array([int(ch_pair.ch1.idx()) for ch_pair in ch_it], dtype=np.uint64)
    cdef cnp.ndarray[cnp.uint64_t, ndim=1] ch2_idx_arr = np.array([int(ch_pair.ch2.idx()) for ch_pair in ch_it], dtype=np.uint64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] result = np.zeros([num_idx, num_fft], dtype=np.float64)

    with nogil:
        for idx in prange(num_idx):
            ch1_idx = ch1_idx_arr[idx]
            ch2_idx = ch2_idx_arr[idx]

            for nn in range(num_fft):
                _tmp = 0.0
                for bb in range(num_bins):
                    _tmp = _tmp + data[ch1_idx, nn, bb] * conj(data[ch2_idx, nn, bb])
                _tmp = _tmp / num_bins
            
                result[idx, nn] = atan2(cimag(_tmp), creal(_tmp)) 

    return(result)

 

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def kernel_crossphase_32_cy(cnp.ndarray[cnp.complex64_t, ndim=3] data, ch_it, fft_config):
    
    cdef size_t num_idx = len(ch_it)      # Length of index array
    cdef size_t num_fft = data.shape[1]   # Number of fft frequencies
    cdef size_t num_bins = data.shape[2]  # Number of ffts
    cdef size_t ch1_idx
    cdef size_t ch2_idx
    cdef size_t idx, nn, bb # Loop variables
    cdef double complex _tmp

    cdef cnp.ndarray[cnp.uint64_t, ndim=1] ch1_idx_arr = np.array([int(ch_pair.ch1.idx()) for ch_pair in ch_it], dtype=np.uint64)
    cdef cnp.ndarray[cnp.uint64_t, ndim=1] ch2_idx_arr = np.array([int(ch_pair.ch2.idx()) for ch_pair in ch_it], dtype=np.uint64)
    cdef cnp.ndarray[cnp.float32_t, ndim=2] result = np.zeros([num_idx, num_fft], dtype=np.float32)

    with nogil:
        for idx in prange(num_idx):
            ch1_idx = ch1_idx_arr[idx]
            ch2_idx = ch2_idx_arr[idx]

            for nn in range(num_fft):
                _tmp = 0.0
                for bb in range(num_bins):
                    _tmp = _tmp + data[ch1_idx, nn, bb] * conj(data[ch2_idx, nn, bb])
                _tmp = _tmp / num_bins
            
                result[idx, nn] = atan2(cimag(_tmp), creal(_tmp)) 

    return(result)


#def kernel_crossphase_64_v2(cnp.ndarray[cnp.complex128_t, ndim=3] data, ch_it, fft_config):
#    cdef size_t num_idx = len(ch_it)      # Length of index array
#    cdef size_t num_fft = data.shape[1]   # Number of fft frequencies
#    cdef size_t num_bins = data.shape[2]  # Number of ffts
#    cdef cnp.ndarray[cnp.uintp_t, ndim=1] ch1_idx_arr = np.array([np.uintp(ch_pair.ch1.idx()) for ch_pair in ch_it], dtype=np.uint64)
#    cdef cnp.ndarray[cnp.uintp_t, ndim=1] ch2_idx_arr = np.array([np.uintp(ch_pair.ch2.idx()) for ch_pair in ch_it], dtype=np.uint64)
#    cdef cnp.ndarray[cnp.float64_t, ndim=2] result = np.zeros([num_idx, num_fft], dtype=np.float64)
#
#    kernel_crossphase_64(&data[0, 0, 0], &result[0, 0], &ch1_idx_arr[0], &ch2_idx_arr[0], num_idx, num_fft, num_bins)
# 
#    return(result)
#
#
#def kernel_crossphase_32_v2(cnp.ndarray[cnp.complex64_t, ndim=3] data, ch_it, fft_config):
#    cdef size_t num_idx = len(ch_it)      # Length of index array
#    cdef size_t num_fft = data.shape[1]   # Number of fft frequencies
#    cdef size_t num_bins = data.shape[2]  # Number of ffts
#    cdef cnp.ndarray[cnp.uintp_t, ndim=1] ch1_idx_arr = np.array([np.uintp(ch_pair.ch1.idx()) for ch_pair in ch_it], dtype=np.uint64)
#    cdef cnp.ndarray[cnp.uintp_t, ndim=1] ch2_idx_arr = np.array([np.uintp(ch_pair.ch2.idx()) for ch_pair in ch_it], dtype=np.uint64)
#    cdef cnp.ndarray[cnp.float32_t, ndim=2] result = np.zeros([num_idx, num_fft], dtype=np.float32)
#
#    kernel_crossphase_32(&data[0, 0, 0], &result[0, 0], &ch1_idx_arr[0], &ch2_idx_arr[0], num_idx, num_fft, num_bins)
# 
#    return(result)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def kernel_crosspower_64_cy(cnp.ndarray[cnp.complex128_t, ndim=3] data, ch_it, fft_config):
    
    cdef size_t num_idx = len(ch_it)      # Length of index array
    cdef size_t num_fft = data.shape[1]   # Number of fft frequencies
    cdef size_t num_bins = data.shape[2]  # Number of ffts
    cdef size_t ch1_idx
    cdef size_t ch2_idx
    cdef size_t idx, nn, bb # Loop variables
    cdef double complex _tmp

    cdef cnp.ndarray[cnp.uint64_t, ndim=1] ch1_idx_arr = np.array([int(ch_pair.ch1.idx()) for ch_pair in ch_it], dtype=np.uint64)
    cdef cnp.ndarray[cnp.uint64_t, ndim=1] ch2_idx_arr = np.array([int(ch_pair.ch2.idx()) for ch_pair in ch_it], dtype=np.uint64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] result = np.zeros([num_idx, num_fft], dtype=np.float64)

    with nogil:
        for idx in prange(num_idx):
            ch1_idx = ch1_idx_arr[idx]
            ch2_idx = ch2_idx_arr[idx]

            for nn in range(num_fft):
                _tmp = 0.0
                for bb in range(num_bins):
                    _tmp = _tmp + data[ch1_idx, nn, bb] * conj(data[ch2_idx, nn, bb])
                    
                result[idx, nn] = cabs(_tmp) / num_bins

    return(result)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def kernel_crosspower_32_cy(cnp.ndarray[cnp.complex64_t, ndim=3] data, ch_it, fft_config):
    
    cdef size_t num_idx = len(ch_it)      # Length of index array
    cdef size_t num_fft = data.shape[1]   # Number of fft frequencies
    cdef size_t num_bins = data.shape[2]  # Number of ffts
    cdef size_t ch1_idx
    cdef size_t ch2_idx
    cdef size_t idx, nn, bb # Loop variables
    cdef double complex _tmp

    cdef cnp.ndarray[cnp.uint64_t, ndim=1] ch1_idx_arr = np.array([int(ch_pair.ch1.idx()) for ch_pair in ch_it], dtype=np.uint64)
    cdef cnp.ndarray[cnp.uint64_t, ndim=1] ch2_idx_arr = np.array([int(ch_pair.ch2.idx()) for ch_pair in ch_it], dtype=np.uint64)
    cdef cnp.ndarray[cnp.float32_t, ndim=2] result = np.zeros([num_idx, num_fft], dtype=np.float32)

    with nogil:
        for idx in prange(num_idx):
            ch1_idx = ch1_idx_arr[idx]
            ch2_idx = ch2_idx_arr[idx]

            for nn in range(num_fft):
                _tmp = 0.0
                for bb in range(num_bins):
                    _tmp = _tmp + data[ch1_idx, nn, bb] * conj(data[ch2_idx, nn, bb])
                    
                result[idx, nn] = cabs(_tmp) / num_bins

    return(result)



#def kernel_crosspower_64_v2(cnp.ndarray[cnp.complex128_t, ndim=3] data, ch_it, fft_config):
#    cdef size_t num_idx = len(ch_it)      # Length of index array
#    cdef size_t num_fft = data.shape[1]   # Number of fft frequencies
#    cdef size_t num_bins = data.shape[2]  # Number of ffts
#    cdef cnp.ndarray[cnp.uintp_t, ndim=1] ch1_idx_arr = np.array([np.uintp(ch_pair.ch1.idx()) for ch_pair in ch_it], dtype=np.uint64)
#    cdef cnp.ndarray[cnp.uintp_t, ndim=1] ch2_idx_arr = np.array([np.uintp(ch_pair.ch2.idx()) for ch_pair in ch_it], dtype=np.uint64)
#    cdef cnp.ndarray[cnp.float64_t, ndim=2] result = np.zeros([num_idx, num_fft], dtype=np.float64)
#
#    kernel_crosspower_64(&data[0, 0, 0], &result[0, 0], &ch1_idx_arr[0], &ch2_idx_arr[0], num_idx, num_fft, num_bins)
# 
#    return(result)
#
#
#def kernel_crosspower_32_v2(cnp.ndarray[cnp.complex64_t, ndim=3] data, ch_it, fft_config):
#    cdef size_t num_idx = len(ch_it)      # Length of index array
#    cdef size_t num_fft = data.shape[1]   # Number of fft frequencies
#    cdef size_t num_bins = data.shape[2]  # Number of ffts
#    cdef cnp.ndarray[cnp.uintp_t, ndim=1] ch1_idx_arr = np.array([np.uintp(ch_pair.ch1.idx()) for ch_pair in ch_it], dtype=np.uint64)
#    cdef cnp.ndarray[cnp.uintp_t, ndim=1] ch2_idx_arr = np.array([np.uintp(ch_pair.ch2.idx()) for ch_pair in ch_it], dtype=np.uint64)
#    cdef cnp.ndarray[cnp.float32_t, ndim=2] result = np.zeros([num_idx, num_fft], dtype=np.float32)
#
#    kernel_crosspower_32(&data[0, 0, 0], &result[0, 0], &ch1_idx_arr[0], &ch2_idx_arr[0], num_idx, num_fft, num_bins)
# 
#    return(result)

# End of file kernels_spectral_py.pyx
