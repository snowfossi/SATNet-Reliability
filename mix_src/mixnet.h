//
// Created by JP.G on 2023/4/16.
//

#ifndef MIXNET_MIXNET_H
#define MIXNET_MIXNET_H

#endif //MIXNET_MIXNET_H

#include <cstdint>
#include <cstdio>

typedef struct mix_t {
    int b, n, k;
    int32_t *is_input;  // b*n
    int32_t *index;     // b*n
    int32_t *niter;     // b
    float *C;            // n*m
    float *dC;          // b*n*m
    float *z, *dz;      // b*n
    float *V, *U;       // b*n*k
    float *gnrm;        // b*n
    float *cache;
    float *tmp;
} mix_t ;


void mix_init_launcher_cpu(mix_t mix, int32_t *perm);

void mix_forward_launcher_cpu(mix_t mix, int max_iter, float eps);

void mix_backward_launcher_cpu(mix_t mix, float prox_lam);

void dbgout1D(const char* before, const float* A, int len, const char* end);

void dbgout2D(const char* before, const float* A, int R, int C, const char* end);