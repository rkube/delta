#include <stdio.h>
#include <complex.h>
#include <math.h>

#include "kernels.h"


void kernel_coherence(double complex* fft_data,
                      double* result,
                      size_t* ch1_idx_arr,
                      size_t* ch2_idx_arr,
                      size_t num_idx,
                      size_t num_fft,
                      size_t num_bins)
{ 
    double complex tmp = 0.0;
    double complex Pxx = 0.0;
    double complex Pyy = 0.0;
    size_t idx3_ch1;
    size_t idx3_ch2;

    #pragma omp parallel for
    for(size_t cidx = 0; cidx < num_idx; cidx++)
    {
        for(size_t nn = 0; nn < num_fft; nn++)
        {
            tmp = 0.0 + 0.0 * I;
            #pragma omp simd 
            for(size_t bb = 0; bb < num_bins; bb++)
            {
                idx3_ch1 = bb + num_bins * (nn + num_fft * ch1_idx_arr[cidx]);
                idx3_ch2 = bb + num_bins * (nn + num_fft * ch2_idx_arr[cidx]);

                Pxx = fft_data[idx3_ch1] * conj(fft_data[idx3_ch1]);
                Pyy = fft_data[idx3_ch2] * conj(fft_data[idx3_ch2]);
                tmp = tmp + fft_data[idx3_ch1] * conj(fft_data[idx3_ch2]) / csqrt(Pxx * Pyy);
            }
            result[nn + num_fft * cidx] = creal(cabs(tmp)) / (double) num_bins;
        }
    }
}


void kernel_crossphase(double complex* fft_data,
                      double* result,
                      size_t* ch1_idx_arr,
                      size_t* ch2_idx_arr,
                      size_t num_idx,
                      size_t num_fft,
                      size_t num_bins)
{ 
    double complex tmp = 0.0;
    double complex Pxx = 0.0;
    double complex Pyy = 0.0;
    size_t idx3_ch1;
    size_t idx3_ch2;

    #pragma omp parallel for
    for(size_t cidx = 0; cidx < num_idx; cidx++)
    {
        for(size_t nn = 0; nn < num_fft; nn++)
        {
            tmp = 0.0 + 0.0 * I;
            #pragma omp simd 
            for(size_t bb = 0; bb < num_bins; bb++)
            {
                
                idx3_ch1 = bb + num_bins * (nn + num_fft * ch1_idx_arr[cidx]);
                idx3_ch2 = bb + num_bins * (nn + num_fft * ch2_idx_arr[cidx]);
                tmp = tmp + fft_data[idx3_ch1] * conj(fft_data[idx3_ch2]);
            }
            tmp = tmp / (double) num_bins;
            result[nn + num_fft * cidx] = atan2(cimag(tmp), creal(tmp));
        }
    }
}


void kernel_crosspower(double complex* fft_data,
                       double* result,
                       size_t* ch1_idx_arr,
                       size_t* ch2_idx_arr,
                       size_t num_idx,
                       size_t num_fft,
                       size_t num_bins)
{ 
    double complex tmp = 0.0;
    double complex Pxx = 0.0;
    double complex Pyy = 0.0;
    size_t idx3_ch1;
    size_t idx3_ch2;

    #pragma omp parallel for
    for(size_t cidx = 0; cidx < num_idx; cidx++)
    {
        for(size_t nn = 0; nn < num_fft; nn++)
        {
            tmp = 0.0 + 0.0 * I;
            #pragma omp simd 
            for(size_t bb = 0; bb < num_bins; bb++)
            {
                
                idx3_ch1 = bb + num_bins * (nn + num_fft * ch1_idx_arr[cidx]);
                idx3_ch2 = bb + num_bins * (nn + num_fft * ch2_idx_arr[cidx]);
                tmp = tmp + fft_data[idx3_ch1] * conj(fft_data[idx3_ch2]);
            }
            
            result[nn + num_fft * cidx] = cabs(tmp) / (double) num_bins;
        }
    }
}