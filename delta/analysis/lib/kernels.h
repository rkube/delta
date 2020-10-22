#include <complex.h>
#include <stdint.h>


void kernel_coherence_64(double complex*, double*, size_t*, size_t*, size_t, size_t, size_t);

void kernel_coherence_32(float complex*, float*, size_t*, size_t*, size_t, size_t, size_t);

void kernel_crossphase_64(double complex*, double*, size_t*, size_t*, size_t, size_t, size_t);

void kernel_crossphase_32(float complex*, float*, size_t*, size_t*, size_t, size_t, size_t);

void kernel_crosspower_64(double complex*, double*, size_t*, size_t*, size_t, size_t, size_t);

void kernel_crosspower_32(float complex*, float*, size_t*, size_t*, size_t, size_t, size_t);