#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include <xmmintrin.h>
#include <smmintrin.h>
#include <float.h>
#include <vector>

#include <omp.h>

#define maxpy mymaxpy
#define mcopy mymcopy
#define mscal mymscal
#define mdot  mymdot
#define mnrm2 mymnrm2
#define mzero mymzero
#define saturate mysaturate

#include "mixnet.h"

const double MEPS = 1e-24;

void maxpy(float* y, float a, const float* x, int l) {
    for (int i = 0; i < l; ++i) {
        y[i] += a * x[i];
    }
}

void mcopy(float *x, float *y, int l)
{
    memcpy(y, x, sizeof(*x)*(size_t)l);
}

float mdot (const float* x, const float* y, int l) {
    float res = 0.0;
    for (int i = 0; i < l; ++i) {
        res += x[i] * y[i];
    }
    return res;
//    x = (float*)__builtin_assume_aligned(x, 4*sizeof(float));
//    y = (float*)__builtin_assume_aligned(y, 4*sizeof(float));
//    __m128 s = _mm_set1_ps(0);
//    for(int i=0; i<l; i+=4, x+=4, y+=4){
//        __m128 x_ = _mm_load_ps(x);
//        __m128 y_ = _mm_load_ps(y);
//        __m128 t = _mm_dp_ps(x_, y_, 0xf1);
//        s = _mm_add_ss(s, t);
//    }
//    float s_;
//    _mm_store_ss(&s_, s);
//
//    return s_;
}

void mscal(float *x, float a, int l)
{
    int m = l-4;
    int i;
    for (i = 0; i < m; i += 5){
        x[i] *= a;
        x[i+1] *= a;
        x[i+2] *= a;
        x[i+3] *= a;
        x[i+4] *= a;
    }

    for ( ; i < l; i++)        /* clean-up loop */
        x[i] *= a;
}

float mnrm2(const float *x, int l)
{
    float xx = mdot(x, x, l);
    return sqrt(xx);
}

void mzero(float *v, int l)
{
    memset(v, 0, l*sizeof(*v));
}

inline float saturate(float x)
{
    return x - (x<0)*x + (x>1)*(1-x);
}


void dbgout1D(const char* before, const float* A, int len, const char* end) {
    printf("%s", before);
    for (int i=0; i<len; i++){
        printf("%7.4f ", A[i]);
    }
    printf("%s", end);
}

void dbgout2D(const char* before, const float* A, int R, int C, const char* end) {
    printf("%s", before);
    for(int i = 0; i < R; ++i) {
        dbgout1D("", A + i * C, C, "\n");
    }
    printf("%s", end);
}

void mix_init(int32_t *perm, int n, int k, const int32_t *is_input, int32_t *index, const float *z, float *V, float *tmp)
{
    //rand_unit(V+0, k);
    // normalize truth-vector
    float s = mdot(V, V, k);
    s = 1/sqrtf(s);
    mscal(V, s, k);

    // printf("in mix_init:\n");
    for (int i=0; i<n; i++) {
        // printf("i=%d  is_input: %s\n", i, is_input[i] ? "Yes" : "No");
        if (is_input[i]) {
            //float Vi1 = V[i*k+1];
            //szero(V+i*k, k);
            //V[i*k] = -cos(z[i]*M_PI);
            //V[i*k+1] = copysign(sin(z[i]*M_PI), Vi1);
            if (i > 0) {
                // initialization with Equation (5)
                const float a = -cos(z[i]*M_PI);
                const float b = sin(z[i]*M_PI);
//                std::vector<float> tv;
                for (int j = 0; j < k; ++j) {
//                    float r = 0.0;
                    tmp[j] = 0;
                    for (int d = 0; d < k; ++d) {
                        if (j == d) {
                            tmp[j] += (1.0 - V[j] * V[d]) * V[i*k + d];
                        }
                        else {
                            tmp[j] += -V[j] * V[d] * V[i*k + d];
                        }
                    }
                    tmp[j] *= b;
                    tmp[j] += a * V[j];
                }
                for (int j = 0; j < k; ++j) {
                    V[i*k + j] = tmp[j];
                }
            }
        }
        // always normalize
        float ss = mdot(V+i*k, V+i*k, k);
        ss = 1/sqrtf(ss);
        mscal(V+i*k, ss, k);

    }
    int i_=0, j=0;
    for (; i_<n-1; i_++) {
        int i = perm[i_]+1;
        if (!is_input[i]) index[j++] = i;
    }
    for (; j<n; j++) index[j] = 0;
}

float mix_kernel_forward(int n, int k, const int32_t *__restrict__ index, const float *__restrict__ C, float *__restrict__ V, float *__restrict__ gnrm, float *__restrict__ g, float *__restrict__ tmp)
{
    float delta = 0;
    for (int i, i_=0; (i=index[i_]); i_++) {
        for (int kk=0; kk<k; kk++){
            g[kk] = 0;
            for (int j=0; j<n; j++)
               g[kk] += C[i * n + j] * V[j * k + kk];
        }

        float gnrmi = mnrm2(g, k);
//        printf("%f\n", gnrmi);
//        if (!gnrmi) mscal(g, 0, k);
//        else mscal(g, -1/gnrmi, k);
        mscal(g, -1/gnrmi, k);

        // dbgout1D("after rescaling by dividing gnrmi, g:\n", g, k);

        float t;
        for (int kk=0; kk<k; kk++){
            t = g[kk], g[kk] -= V[i*k+kk], V[i*k+kk] = t;
        }

        // Calc function decrease, i.e., gnorm * (vi_new - vi_old)^2
        const float dec =  gnrmi * mdot(g, g, k);
        delta += dec;
        gnrm[i] = gnrmi;
        //printf("coordinate update on dimention %d, dec: %f, accumulated delta: %f, gnrmi: %f\n", i, dec, delta, gnrmi);
    }
    return delta; // only useful for the forward pass
}

void mix_kernel_backward(int n, int k, float prox_lam, int32_t * is_input, const int32_t *__restrict__ index, const float *__restrict__ C,
                          const float *__restrict__ z, const float *__restrict__ dz,const float *__restrict__ V, float *__restrict__ U, float *__restrict__ gnrm, float *__restrict__ tmp)
{
    for (int i, i_=0; (i=index[i_]); i_++) {
        float gnrmi = gnrm[i] + prox_lam;
        // dvo - \sum_j coj*uj
        for (int d=0; d<k; d++){
            tmp[d] = 0;
            for (int j=0; j<n; ++j)
                if(!is_input[j]){
                    tmp[d] -= C[i * n + j] * U[j * k + d];
                }
//            tu[d] = -r;
            float zi = z[i];
            if (zi > 0.95) {
                zi = 0.95;
            }
            else if (zi < 0.05) {
                zi = 0.05;
            }
            float dvi = dz[i]/M_PI/sin(zi*M_PI) * V[d];
            tmp[d] += dvi;
        }

        // P_o = I_k - v_o * v'_o
        //tu = (dvo - \sum coj uj)
        // u_o = 1 / g_nrmo * P_o * tu = 1/ g_nrmo * (tu - v_o (v_o' * tu))
        //tmp[d] (dvo - \sum coj uj)
        for (int d=0; d<k; d++){
            tmp[k + d] = 0;
            for (int kk=0; kk<k; kk++){
                tmp[k + d] += V[i * k + kk] * tmp[kk];      // (v_o' * tu)
            }
            tmp[k + d] = tmp[d] - tmp[k + d] * V[i * k + d];
//            if (fabs(U[i*k+d] - tmp[k + d] / gnrmi) > 0.05)
//                printf("U=%f, delta=%f\n", U[i*k+d], fabs(U[i*k+d] - tmp[k + d] / gnrmi));
//            if (fabs(U[i*k+d] - tmp[k + d] / gnrmi) > 20) {
//                printf("U_%d %d does not converge, U=%f, delta=%f\n", i, d, U[i*k+d], fabs(U[i*k+d] - tmp[k + d] / gnrmi));
//                continue;
//            }
            U[i*k+d] = tmp[k + d] / gnrmi;
            if (U[i * k + d] > 100 || U[i * k + d] < -100) {
//                printf("U_%d %d too large!, U=%f\n", i, d, U[i*k+d]);
                U[i*k+d] = 0;
            }
        }
    }
}

void mix_forward(int max_iter, float eps, int n, int k, const int32_t *index, int32_t *niter, const float *C, float *z, float *V, float *gnrm, float *cache, float *tmp)
{
    float delta;
    int iter = 0;
    for (; iter < max_iter; iter++) {
        delta = mix_kernel_forward(n, k, index, C, V, gnrm, cache, tmp);
        if (iter && delta < eps) break;
        if (iter == 0) eps = delta*eps;
    }

    *niter = iter;
//    *niter = max_iter;
    for (int i,i_=0; (i=index[i_]); i_++) {
        float zi = mdot(V, V+i*k, k);
        zi = saturate((zi+1)/2)*2-1;
        zi = saturate(1-acosf(zi)/M_PI);
        z[i] = zi;
    }
}

void mix_backward(float prox_lam, int n, int k, int32_t *is_input, int32_t *index, int32_t *niter, const float *C, float *dC,
                  float *z, float *dz, const float *V, float *U, float *gnrm, float *tmp)
{
    //printf("mix_backward:\n");
    int invalid_flag=0;
    for (int i,i_=0; (i=index[i_]); i_++) {
        float zi = z[i];
        // Equation (8)
        if (zi > 0.95) {
            zi = 0.95;
        }
        if (zi < 0.05) {
            zi = 0.05;
        }
        float dzi = dz[i]/M_PI/sin(zi*M_PI);

        if (isnan(dzi) || isinf(dzi)) {
            printf("gnrm[%d] = %f\n", i, gnrm[i]);
            invalid_flag = 1;
        }
    }
    if (invalid_flag) { mzero(dz, n); return; }
    //compute gnrm
    for (int i=0; i<n; ++i){
        for (int kk=0; kk<k; kk++){
            tmp[kk] = 0;
            for (int j=0; j<n; j++)
               tmp[kk] += C[i * n + j] * V[j * k + kk];
        }

        float gnrmi = 0;
        for (int d=0; d<k; d++)
            gnrmi += tmp[d] * tmp[d];
        gnrm[i] = sqrt(gnrmi);
        if (gnrm[i] < MEPS) {
            printf("gnrm too small. gnrm[%d] = %f\n", i, gnrm[i]);
            invalid_flag = 1;
        }
    }
    if (invalid_flag) { mzero(dz, n); return; }

//    dbgout2D("U after mix:\n", U, n, k, "\n");
    // solve P (S'S+D_z-D_sii)xI_k P U = -dz P v0
//    printf("%d\n", *niter);
    for (int iter=0; iter<*niter; iter++) {
        mix_kernel_backward(n, k, prox_lam, is_input, index, C, z, dz, V, U, gnrm, tmp);
    }
//    dbgout2D("U before mix:\n", U, n, k, "\n");
    // sanity check
    for (int ik=0; ik<n*k; ik++) {
        if (isnan(U[ik]) || isinf(U[ik])) invalid_flag = 1;
    }
    if (invalid_flag) { mzero(dz, n); return; }

    // Note that the gradient sign is reversed here
    // dC_ij = -ui'*vj - uj'vi
    for (int i=0; i<n; i++){
        for (int j=0; j<n; j++){
            if(i==j) dC[i * n + j] = 0;
            else if (C[i * n + j] == 0) dC[i * n + j] = 0; //may not hold mathematically
            else dC[i * n + j] = -mdot(U + i * k, V + j * k, k) -mdot(U + j * k, V + i * k, k); // enforce symmetry
//            else dC[i * n + j] = -mdot(U + i * k, V + j * k, k);
        }
    }

    // dzi = v0'Phi si
    for (int i=1; i<n; i++) {
        if (!is_input[i]) {
            dz[i] = 0;
        }

        // Equation (12) and (13), the latter is similar to Equation (5)
        const float a = sin(z[i]*M_PI);
        const float b = cos(z[i]*M_PI);
        // compute  (I_k - V_true V'_true) V_i_rand
        for (int j = 0; j < k; ++j) {
//            float r = 0.0;
            tmp[j] = 0;
            for (int d = 0; d < k; ++d) {
                if (j == d) {
//                    r += (1.0 - V[j] * V[d]) * V[i*k + d];
                    tmp[j] += (1.0 - V[j] * V[d]) * V[i*k + d];
                }
                else {
//                    r += -V[j] * V[d] * V[i*k + d];
                    tmp[j] += -V[j] * V[d] * V[i*k + d];
                }
            }
//            r *= b;
//            r += a * V[j];
//            r *= M_PI;
//            tv.push_back(r);
            tmp[j] *= b;
            tmp[j] += a * V[j];
            tmp[j] *= M_PI;
        }

        for (int d=0; d<k; ++d){
            tmp[k + d] = 0;
            for (int ii=0; ii<n; ++ii)
                if (!is_input[ii])
                    tmp[k + d] -= U[ii * k + d] * C[ii * n + i];
        }
        for (int j = 0; j<k; ++j) {
            dz[i] -= tmp[j] * tmp[k + j];
        }
    }
}

void mix_init_launcher_cpu(mix_t mix, int32_t *perm)
{
    int n=mix.n, k=mix.k;
#pragma omp parallel for schedule(dynamic)
    for (int i=0; i<mix.b; i++) {
        mix_init(perm,
                 mix.n, mix.k, mix.is_input+i*n, mix.index+i*n, mix.z+i*n,
                 mix.V+i*n*k, mix.tmp+i*n*k);
    }
}

void mix_forward_launcher_cpu(mix_t mix, int max_iter, float eps)
{
    int n=mix.n, k=mix.k;
#pragma omp parallel for schedule(dynamic)
    for (int i=0; i<mix.b; i++) {
        mix_forward(max_iter, eps,
                    mix.n,mix.k, mix.index+i*n, mix.niter+i,
                    mix.C, mix.z+i*n, mix.V+i*n*k, mix.gnrm+i*n, mix.cache+i*k, mix.tmp+i*n*k);
        // dbgout2D("after mixing, V:\n", mix.V + i * n * k, n, k);
    }
}

void mix_backward_launcher_cpu(mix_t mix, float prox_lam)
{
    int n=mix.n, k=mix.k;
#pragma omp parallel for schedule(dynamic)
    for (int i=0; i<mix.b; i++) {
        mix_backward(prox_lam,
                     mix.n, mix.k, mix.is_input+i*n, mix.index+i*n, mix.niter+i,
                     mix.C, mix.dC+i*n*n, mix.z+i*n, mix.dz+i*n, mix.V+i*n*k, mix.U+i*n*k,
                      mix.gnrm+i*n, mix.tmp+i*n*k);
    }
}
