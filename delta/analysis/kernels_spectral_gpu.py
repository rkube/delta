# -*- Encoding: UTF-8 -*-

from numba import jit, cuda


@cuda.jit
def kernel_spectral_GAP(in_arr, out_arr, ch1_idx_arr, ch2_idx_arr, win_factor=0.0):
    """Calculates G, A, P in one call.

    G: coherence
    A: cross-phase
    P: cross-power

    Arguments:
        in_arr: ndarray(num_ch, num_fft, num_bins):
            STFT data. axis 0: channel, axis 1: Fourier Coefficient, axis 2: Estimate of
                Fourier Coefficient for a given channel
        out_arr: ndarray(num_pairs, num_fft, 3):
            axis 0: channel pair. axis 1: Fourier Coefficient. axis 2: [G, P, S]
        ch1_idx_arr: ndarray(num_pairs)
            First channel of the channel pairs
        ch2_idx_arr: ndarray(num_pairs)
            Second channel of the channel pairs
        win_factor: float
            Optional windowing factor.

    Returns:
        None
    """
    # Obtain position in the output array from the launch configuration
    pair_idx, nn = cuda.grid(2)
    # Infer indices to be used to calculate output quantite
    ch1_idx = ch1_idx_arr[pair_idx]
    ch2_idx = ch2_idx_arr[pair_idx]

    if pair_idx < out_arr.shape[0] and nn < out_arr.shape[1]:
        # G: coherence
        G = in_arr[ch1_idx, nn, 0] - in_arr[ch1_idx, nn, 0] # G: coherence
        # A: cross-phase, P: cross-power
        # They are calculated from the same sum
        AP = in_arr[ch1_idx, nn, 0] - in_arr[ch1_idx, nn, 0] # A: cross-phase and cross-phase
        for k in range(in_arr.shape[2]):
            X = in_arr[ch1_idx, nn, k]
            Y = in_arr[ch2_idx, nn, k]
            Pxx = X * X.conjugate()
            Pyy = Y * Y.conjugate()
            Pxy = X * Y.conjugate()

            G += Pxy / sqrt(Pxx * Pyy)
            AP += Pxy

        G /= in_arr.shape[2]
        AP /= in_arr.shape[2]
        
        # Write coherence to output array
        out_arr[pair_idx, nn, 0] = abs(G)
        out_arr[pair_idx, nn, 1] = arctan2(AP.imag, AP.real)
        out_arr[pair_idx, nn, 2] = abs(AP).real / win_factor


# End of file kernels_spectral_gpu.py