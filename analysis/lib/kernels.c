#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>

#include "kernels.h"


void kernel_coherence_64(double complex* __restrict__ fft_data,
                         double* __restrict__ result,
                         size_t* __restrict__ ch1_idx_arr,
                         size_t* __restrict__ ch2_idx_arr,
                         size_t num_idx,
                         size_t num_fft,
                         size_t num_bins)
{ 
    double complex tmp = 0.0;
    double complex Pxx = 0.0;
    double complex Pyy = 0.0;
    size_t idx3_ch1 = 0;
    size_t idx3_ch2 = 0;
    size_t nn_num_bins = 0;

    #pragma omp parallel for collapse(2)
    for(size_t cidx = 0; cidx < num_idx; cidx++)
    {
        // The correct index is idx3_ch1 = bb + num_bins * (nn + num_fft * ch1_idx_arr[cidx]);
        // Factoring out this expression, we can divide this one expression that varies with the outer loop
        // 
        for(size_t nn = 0; nn < num_fft; nn++)
        {
            tmp = 0.0 + 0.0 * I;
            idx3_ch1 = num_bins * num_fft * ch1_idx_arr[cidx] + nn * num_bins;
            idx3_ch2 = num_bins * num_fft * ch1_idx_arr[cidx] + nn * num_bins;
            #pragma omp simd reduction(+: tmp)
            for(size_t bb = 0; bb < num_bins; bb++)
            {
                Pxx = fft_data[idx3_ch1 +  bb] * conj(fft_data[idx3_ch1 +  bb]);
                Pyy = fft_data[idx3_ch2 +  bb] * conj(fft_data[idx3_ch2 +  bb]);
                tmp = tmp + fft_data[idx3_ch1 +  bb] * conj(fft_data[idx3_ch2 +  bb]) / csqrt(Pxx * Pyy);
            }
            result[nn + num_fft * cidx] = creal(cabs(tmp)) / (double) num_bins;
        }
    }
}


void kernel_coherence_32(float complex* __restrict__ fft_data,
                         float* __restrict__ result,
                         size_t* __restrict__ ch1_idx_arr,
                         size_t* __restrict__ ch2_idx_arr,
                         size_t num_idx,
                         size_t num_fft,
                         size_t num_bins)
{ 
    float complex tmp = 0.0;
    float complex Pxx = 0.0;
    float complex Pyy = 0.0;
    size_t idx3_ch1 = 0;
    size_t idx3_ch2 = 0;

    #pragma omp parallel for collapse(2)
    for(size_t cidx = 0; cidx < num_idx; cidx++)
    {
        // The correct index is idx3_ch1 = bb + num_bins * (nn + num_fft * ch1_idx_arr[cidx]);
        // Factoring out this expression, we can divide this one expression that varies with the outer loop
        // 
        for(size_t nn = 0; nn < num_fft; nn++)
        {
            tmp = 0.0 + 0.0 * I;
            idx3_ch1 = num_bins * (nn + num_fft * ch1_idx_arr[cidx]);
            idx3_ch2 = num_bins * (nn + num_fft * ch2_idx_arr[cidx]);

            #pragma omp simd reduction(+: tmp) 
            for(size_t bb = 0; bb < num_bins; bb++)
            {
                Pxx = fft_data[idx3_ch1 + bb] * conj(fft_data[idx3_ch1 + bb]);
                Pyy = fft_data[idx3_ch2 + bb] * conj(fft_data[idx3_ch2 + bb]);
                tmp = tmp + fft_data[idx3_ch1 + bb] * conj(fft_data[idx3_ch2 + bb]) / csqrt(Pxx * Pyy);
            }

            result[nn + num_fft * cidx] = creal(cabs(tmp)) / (float) num_bins;
        }
    }
}


void kernel_crossphase_64(double complex* __restrict__ fft_data,
                          double* __restrict__ result,
                          size_t* __restrict__ ch1_idx_arr,
                          size_t* __restrict__ ch2_idx_arr,
                          size_t num_idx,
                          size_t num_fft,
                          size_t num_bins)
{ 
    double complex tmp = 0.0;
    double complex Pxx = 0.0;
    double complex Pyy = 0.0;
    size_t idx3_ch1;
    size_t idx3_ch2;

    #pragma omp parallel for collapse(2)
    for(size_t cidx = 0; cidx < num_idx; cidx++)
    {
        for(size_t nn = 0; nn < num_fft; nn++)
        {
            tmp = 0.0 + 0.0 * I;
            idx3_ch1 = num_bins * (nn + num_fft * ch1_idx_arr[cidx]);
            idx3_ch2 = num_bins * (nn + num_fft * ch2_idx_arr[cidx]);
            #pragma omp simd reduction(+: tmp)
            for(size_t bb = 0; bb < num_bins; bb++)
            {
                
                tmp = tmp + fft_data[idx3_ch1 + bb] * conj(fft_data[idx3_ch2 + bb]);
            }
            tmp = tmp / (double) num_bins;
            result[nn + num_fft * cidx] = atan2(cimag(tmp), creal(tmp));
        }
    }
}


void kernel_crossphase_32(float complex* __restrict__ fft_data,
                          float* __restrict__ result,
                          size_t* __restrict__ ch1_idx_arr,
                          size_t* __restrict__ ch2_idx_arr,
                          size_t num_idx,
                          size_t num_fft,
                          size_t num_bins)
{ 
    float complex tmp = 0.0;
    float complex Pxx = 0.0;
    float complex Pyy = 0.0;
    size_t idx3_ch1;
    size_t idx3_ch2;

    #pragma omp parallel for collapse(2)
    for(size_t cidx = 0; cidx < num_idx; cidx++)
    {
        for(size_t nn = 0; nn < num_fft; nn++)
        {
            tmp = 0.0 + 0.0 * I;
            idx3_ch1 = num_bins * (nn + num_fft * ch1_idx_arr[cidx]);
            idx3_ch2 = num_bins * (nn + num_fft * ch2_idx_arr[cidx]);
            #pragma omp simd reduction(+: tmp)
            for(size_t bb = 0; bb < num_bins; bb++)
            {
                
                tmp = tmp + fft_data[idx3_ch1 + bb] * conj(fft_data[idx3_ch2 + bb]);
            }
            tmp = tmp / (float) num_bins;
            result[nn + num_fft * cidx] = atan2(cimag(tmp), creal(tmp));
        }
    }
}


void kernel_crosspower_64(double complex* __restrict__ fft_data,
                          double* __restrict__ result,
                          size_t* __restrict__ ch1_idx_arr,
                          size_t* __restrict__ ch2_idx_arr,
                          size_t num_idx,
                          size_t num_fft,
                          size_t num_bins)
{ 
    double complex tmp = 0.0;
    double complex Pxx = 0.0;
    double complex Pyy = 0.0;
    size_t idx3_ch1;
    size_t idx3_ch2;

    #pragma omp parallel for collapse(2)
    for(size_t cidx = 0; cidx < num_idx; cidx++)
    {
        for(size_t nn = 0; nn < num_fft; nn++)
        {
            tmp = 0.0 + 0.0 * I;
            idx3_ch1 = num_bins * (nn + num_fft * ch1_idx_arr[cidx]);
            idx3_ch2 = num_bins * (nn + num_fft * ch2_idx_arr[cidx]);
            #pragma omp simd reduction(+: tmp)
            for(size_t bb = 0; bb < num_bins; bb++)
            {               
                tmp = tmp + fft_data[idx3_ch1 + bb] * conj(fft_data[idx3_ch2 + bb]);
            } 
            result[nn + num_fft * cidx] = cabs(tmp) / (double) num_bins;
        }
    }
}


void kernel_crosspower_32(float complex* __restrict__ fft_data,
                          float* __restrict__ result,
                          size_t* __restrict__ ch1_idx_arr,
                          size_t* __restrict__ ch2_idx_arr,
                          size_t num_idx,
                          size_t num_fft,
                          size_t num_bins)
{ 
    float complex tmp = 0.0;
    float complex Pxx = 0.0;
    float complex Pyy = 0.0;
    size_t idx3_ch1;
    size_t idx3_ch2;

    #pragma omp parallel for collapse(2)
    for(size_t cidx = 0; cidx < num_idx; cidx++)
    {
        for(size_t nn = 0; nn < num_fft; nn++)
        {
            tmp = 0.0 + 0.0 * I;
            idx3_ch1 = num_bins * (nn + num_fft * ch1_idx_arr[cidx]);
            idx3_ch2 = num_bins * (nn + num_fft * ch2_idx_arr[cidx]);
            #pragma omp simd reduction(+: tmp)
            for(size_t bb = 0; bb < num_bins; bb++)
            {               
                tmp = tmp + fft_data[idx3_ch1 + bb] * conj(fft_data[idx3_ch2 + bb]);
            } 
            result[nn + num_fft * cidx] = cabs(tmp) / (float) num_bins;
        }
    }
}

// End of file kernels.c