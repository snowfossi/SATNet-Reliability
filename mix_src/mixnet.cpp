#ifdef MIX_USE_GPU
#include <ATen/cuda/CUDAContext.h>
#endif
#include <torch/extension.h>

#ifdef MIX_USE_GPU
#define DEVICE_NAME cuda
	#define _MIX_DEV_STR "cuda"
	#define _MIX_CUDA_DECL , cudaStream_t stream
	#define _MIX_CUDA_ARG , stream
	#define _MIX_CUDA_HEAD cudaStream_t stream = at::cuda::getCurrentCUDAStream();
	#define _MIX_CUDA_TAIL  AT_CUDA_CHECK(cudaGetLastError());
                            //AT_CUDA_CHECK(cudaStreamSynchronize(stream));
#else
#define DEVICE_NAME cpu
#define _MIX_DEV_STR "cpu"
#define _MIX_CUDA_DECL
#define _MIX_CUDA_ARG
#define _MIX_CUDA_HEAD
#define _MIX_CUDA_TAIL
#endif

// name mangling for CPU and CUDA
#define _MIX_CAT(x,y) x ## _ ## y
#define _MIX_EVAL(x,y) _MIX_CAT(x,y)
#define _MIX_FUNC(name) _MIX_EVAL(name, DEVICE_NAME)

#include "mixnet.h"

using Tensor=torch::Tensor;
float *fptr(Tensor& a) { return a.data_ptr<float>(); }
int   *iptr(Tensor& a) { return a.data_ptr<int>(); }

void _MIX_FUNC(mix_init_launcher)    (mix_t mix, int32_t *perm             _MIX_CUDA_DECL);
void _MIX_FUNC(mix_forward_launcher) (mix_t mix, int max_iter, float eps   _MIX_CUDA_DECL);
void _MIX_FUNC(mix_backward_launcher)(mix_t mix, float prox_lam            _MIX_CUDA_DECL);

void mix_init(Tensor perm,
              Tensor is_input, Tensor index, Tensor z, Tensor V, Tensor tmp)
{
    _MIX_CUDA_HEAD;

    mix_t mix;
    mix.b = V.size(0); mix.n = V.size(1); mix.k = V.size(2);
    mix.is_input = iptr(is_input);
    mix.index = iptr(index);
    mix.z = fptr(z);
    mix.V = fptr(V);
    mix.tmp = fptr(tmp);

    _MIX_FUNC(mix_init_launcher)(mix, iptr(perm) _MIX_CUDA_ARG);

    _MIX_CUDA_TAIL;
}

void mix_forward(int max_iter, float eps,
                 Tensor index, Tensor niter, Tensor C, Tensor z, Tensor V, Tensor gnrm, Tensor cache, Tensor tmp)
{
    _MIX_CUDA_HEAD;

    mix_t mix;
    mix.b = V.size(0); mix.n = V.size(1); mix.k = V.size(2);
    mix.index = iptr(index);
    mix.niter = iptr(niter);
    mix.C = fptr(C);
    mix.z = fptr(z);
    mix.V = fptr(V);
    mix.gnrm = fptr(gnrm);
    mix.cache = fptr(cache);
    mix.tmp = fptr(tmp);

    _MIX_FUNC(mix_forward_launcher)(mix, max_iter, eps _MIX_CUDA_ARG);

    _MIX_CUDA_TAIL;
}

void mix_backward(float prox_lam,
                  Tensor is_input, Tensor index, Tensor niter, Tensor C, Tensor dC, Tensor z, Tensor dz,
                  Tensor V, Tensor U, Tensor gnrm, Tensor cache, Tensor tmp)
{
    _MIX_CUDA_HEAD;

    mix_t mix;
    mix.b = V.size(0); mix.n = V.size(1); mix.k = V.size(2);
    mix.is_input = iptr(is_input);
    mix.index = iptr(index);
    mix.niter = iptr(niter);
    mix.C = fptr(C); mix.dC = fptr(dC);
    mix.z = fptr(z); mix.dz = fptr(dz);
    mix.V = fptr(V); mix.U = fptr(U);
    mix.gnrm = fptr(gnrm);
    mix.cache = fptr(cache);
    mix.tmp = fptr(tmp);

    _MIX_FUNC(mix_backward_launcher)(mix, prox_lam _MIX_CUDA_ARG);

    _MIX_CUDA_TAIL;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("init" , &mix_init, "MIXNet init (" _MIX_DEV_STR ")");
m.def("forward" , &mix_forward, "MIXNet forward (" _MIX_DEV_STR ")");
m.def("backward" , &mix_backward, "MIXNet backward (" _MIX_DEV_STR ")");
}