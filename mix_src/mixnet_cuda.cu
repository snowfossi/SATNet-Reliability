#include <math.h>
#include <stdint.h>
#include <float.h>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include "mixnet.h"

const double MEPS = 1e-24;
const int WARP_SIZE = 32;
const int WARP_NUM = 32;
const int MBUF_SIZE = 320;

// Warp level dot product
__device__ float warpdot(const float * x, const float * z, int k)
{
    if (k==0) return 0;
    int lane = threadIdx.x % WARP_SIZE;

    float val = 0;
    #pragma unroll 2
    for (int i=lane; i<k; i+=WARP_SIZE) val += x[i]*z[i];
    __syncwarp();

    unsigned int active = __activemask();
    #pragma unroll
    for (int off=WARP_SIZE/2; off; off/=2)
        val += __shfl_xor_sync(active, val, off);

    return val;
}

__device__ float warpdot1(const float * x, int shift_x, const float * z, int shift_z, int k)
{
    if (k==0) return 0;
    int lane = threadIdx.x % WARP_SIZE;

    float val = 0;
    #pragma unroll 2
    for (int i=lane; i<k; i+=WARP_SIZE) val += *(x+i*shift_x) * *(z+i*shift_z);
    __syncwarp();

    unsigned int active = __activemask();
    #pragma unroll
    for (int off=WARP_SIZE/2; off; off/=2)
        val += __shfl_xor_sync(active, val, off);

    return val;
}

__device__ float warpdot2(const float * x, int shift_x, const float *y, int shift_y, const int * z, int shift_z, int k)
{
    if (k==0) return 0;
    int lane = threadIdx.x % WARP_SIZE;

    float val = 0;
    #pragma unroll 2
    for (int i=lane; i<k; i+=WARP_SIZE) val += *(x+i*shift_x) * *(y+i*shift_y) * (1 - *(z + i*shift_z));
    __syncwarp();

    unsigned int active = __activemask();
    #pragma unroll
    for (int off=WARP_SIZE/2; off; off/=2)
        val += __shfl_xor_sync(active, val, off);

    return val;
}


__global__ void mix_init(int32_t *perm, int n, int k, const int32_t *is_input, int32_t *index, const float *z, float *V, float *tmp)
{
    is_input +=  n   * blockIdx.x;
    index +=     n   * blockIdx.x;
    z +=         n   * blockIdx.x;
    V +=         n*k * blockIdx.x;
    tmp +=       n*k * blockIdx.x;

    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    for (int i=warp; i<n; i+=WARP_NUM) {
        if (is_input[i]) {
            const float a = -cos(z[i]*M_PI);
            const float b = sin(z[i]*M_PI);
            const float c = warpdot(V, V+i*k, k);
            const float d = a - b * c;

            for (int kk=lane; kk<k; kk+=WARP_SIZE) {
                tmp[i*k+kk] = d * V[kk] + b * V[i*k + kk];
            }
            __syncwarp();
            float s = warpdot(tmp+i*k, tmp+i*k, k);
            s = rsqrtf(s);
            for (int kk=lane; kk<k; kk+=WARP_SIZE) V[i*k+kk] = tmp[i*k + kk] * s;
            __syncwarp();
        }

        float s = warpdot(V+i*k, V+i*k, k);
        s = rsqrtf(s);
        for (int kk=lane; kk<k; kk+=WARP_SIZE) V[i*k+kk] *= s;

    }
    __syncthreads();
    __threadfence_block();


    if (threadIdx.x == 0) {

        int i_=0, j=0;
        for (; i_<n-1; i_++) {
            int i = perm[i_]+1;
            if (!is_input[i]) index[j++] = i;
        }
        for (; j<n; j++) index[j] = 0;
    }
    __syncthreads();
}

__forceinline__
__device__ float mix_kernel_forward(int n, int k, const int32_t *__restrict__ index, const float *__restrict__ C, float *__restrict__ V, float *__restrict__ gnrm, float *__restrict__ g, float *__restrict__ tmp)
{
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    __shared__ float delta;
    if (threadIdx.x == 0) delta = 0;

    for (int i, i_=0; (i=index[i_]); i_++) {
        for (int kk=warp; kk<k; kk+=WARP_NUM) {
            float val = warpdot1(C+i*n, 1, V+kk, k, n);
            __syncwarp();
            if (lane == 0) {
                g[kk] = val;
//                 printf("%f\n", val);
            }
        }
        __syncthreads();

        float gnrmi = sqrtf(warpdot(g, g, k));

        for (int kk=threadIdx.x; kk<k; kk+=blockDim.x) {
            g[kk] /= -gnrmi;
            float t;
            t = g[kk], g[kk] -= V[i*k+kk], V[i*k+kk] = t;
        }
        __syncthreads();

        const float dec =  gnrmi * warpdot(g, g, k);

        if (threadIdx.x == 0) {
            delta += dec;
            gnrm[i] = gnrmi;
        }
        __syncthreads();
    }
    __syncthreads();

    return delta;
}

__global__ void mix_forward(int max_iter, float eps, int n, int k, const int32_t *index, int32_t *niter, const float *C, float *z, float *V, float *gnrm, float *cache, float * tmp)
{
    index +=    n * blockIdx.x;
    z +=        n * blockIdx.x;
    V +=        n*k*blockIdx.x;
    gnrm +=     n * blockIdx.x;
    cache +=    k * blockIdx.x;
    tmp +=      n*k*blockIdx.x;

    float delta;
    int iter = 0;
    for (; iter < max_iter; iter++) {
        delta = mix_kernel_forward(n, k, index, C, V, gnrm, cache, tmp);
        if (iter && delta < eps) break;
        if (iter == 0) eps = delta*eps;
    }

    niter[blockIdx.x] = iter;

    for (int i,i_=0; (i=index[i_]); i_++) {
        float zi = warpdot(V, V+i*k, k);
//         float zi = V[i*k];
        zi = saturate((zi+1)/2)*2-1;
        zi = saturate(1-acosf(zi)/M_PI);
        if (threadIdx.x == 0) z[i] = zi;
    }
}

__forceinline__
__device__ void mix_kernel_backward(int n, int k, float prox_lam, int32_t * is_input, const int32_t *__restrict__ index, const float *__restrict__ C,
                          const float *__restrict__ z, const float *__restrict__ dz,const float *__restrict__ V, float *__restrict__ U, float *__restrict__ gnrm, float  *__restrict__ tmp)
{

    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int NUM = 100000000;

    for (int i, i_=0; (i=index[i_]); i_++) {
        float gnrmi = gnrm[i] + prox_lam;

        for (int d=warp; d<k; d+=WARP_NUM) {
            float val = warpdot2(C+i*n, 1, U+d, k, is_input, 1, n);
            __syncwarp();

            if (lane == 0) {
                tmp[d] = -val + z[i] * V[d];
            }
        }
        __syncthreads();

        float val = warpdot(V+i*k, tmp, k);
        __syncthreads();

        for (int d=threadIdx.x; d<k; d+=blockDim.x) {
            U[i*k+d] = (tmp[d] - val * V[i * k + d]) / gnrmi;
            if (U[i * k + d] > 100 || U[i * k + d] < -100) {
                U[i*k+d] = 0;
            }
        }
        __syncthreads();
    }
}

__global__ void mix_backward(float prox_lam, int n, int k, int32_t *is_input, int32_t *index, int32_t *niter, const float *C, float *dC,
                  float *z, float *dz, const float *V, float *U, float *gnrm, float *tmp)
{
    is_input += n * blockIdx.x;
    index +=    n * blockIdx.x;
    dC +=       n*n*blockIdx.x;
    z +=        n * blockIdx.x;
    dz +=       n * blockIdx.x;
    V +=        n*k*blockIdx.x;
    U +=        n*k*blockIdx.x;
    gnrm +=     n * blockIdx.x;
    tmp +=      n*k*blockIdx.x;

    __shared__ int invalid_flag;
    if (threadIdx.x == 0) invalid_flag = 0;
    // __syncthreads();

    for (int i,i_=0; (i=index[i_]); i_++) {
        if (threadIdx.x == 0) {
            float zi = z[i];

            // Equation (8)
            float dzi = dz[i]/M_PI/sinpif(zi);
            if (isnan(dzi) || isinf(dzi) || gnrm[i] < MEPS) {
                invalid_flag = 1;
            }
            z[i] = dzi;
        }
    }
    __syncthreads();

    if (invalid_flag) {
        for (int i=threadIdx.x; i<n; i+=blockDim.x) dz[i] = 0;
        return;
    }

    // solve P (S'S+D_z-D_sii)xI_k P U = -dz P v0

    for (int iter=0; iter<niter[blockIdx.x]; iter++) {
        mix_kernel_backward(n, k, prox_lam, is_input, index, C, z, dz, V, U, gnrm, tmp);
    }

    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;

    // Note that the gradient sign is reversed here
    // dC_ij = -ui'*vj
    for (int i=warp; i<n; i+=WARP_NUM) {
        for (int j=lane; j<n; j+=WARP_SIZE) {
            if(i==j) dC[i * n + j] = 0;
            else if (C[i * n + j] == 0) dC[i * n + j] = 0;
            else {
                dC[i * n + j] = 0;
                for (int kk=0; kk<k; kk++) {
                    dC[i * n + j] += -U[i*k+kk] * V[j*k+kk] - U[j*k+kk] * V[i*k+kk];
                }
            }
        }

    }
    __syncthreads();
}

void mix_init_launcher_cuda(mix_t mix, int32_t *perm, cudaStream_t stream)
{
    mix_init<<<mix.b,WARP_SIZE*WARP_NUM,0,stream>>>(perm,
        mix.n, mix.k, mix.is_input, mix.index, mix.z, mix.V, mix.tmp);
}

void mix_forward_launcher_cuda(mix_t mix, int max_iter, float eps, cudaStream_t stream)
{
    int smem_size = sizeof(float);
    mix_forward<<<mix.b,WARP_SIZE*WARP_NUM,smem_size,stream>>>(max_iter, eps,
        mix.n, mix.k, mix.index, mix.niter,
        mix.C, mix.z, mix.V, mix.gnrm, mix.cache, mix.tmp);
}

void mix_backward_launcher_cuda(mix_t mix, float prox_lam, cudaStream_t stream)
{
    int smem_size = sizeof(float);

    mix_backward<<<mix.b,WARP_SIZE*WARP_NUM,smem_size,stream>>>(prox_lam,
        mix.n, mix.k, mix.is_input, mix.index, mix.niter,
        mix.C, mix.dC, mix.z, mix.dz, mix.V, mix.U, mix.gnrm, mix.tmp);

    cudaStreamSynchronize(stream);
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << cudaGetErrorString(err) << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        std::exit(EXIT_FAILURE);
    }
}