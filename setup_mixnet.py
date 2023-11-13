import torch.cuda
from os import system
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
from torch.utils.cpp_extension import CUDA_HOME

gencode = [
        '-gencode=arch=compute_50,code=sm_50',
        '-gencode=arch=compute_52,code=sm_52',
        '-gencode=arch=compute_60,code=sm_60',
        '-gencode=arch=compute_61,code=sm_61',
        '-gencode=arch=compute_75,code=sm_75',
        '-gencode=arch=compute_80,code=sm_80',
        '-gencode=arch=compute_86,code=sm_86',
]

ext_modules = [
    CppExtension(
        name = 'mixnet._cpp',
        include_dirs = ['./mix_src'],
        sources = [
            'mix_src/mixnet.cpp',
            'mix_src/mixnet_cpu.cpp',
        ],
        extra_compile_args = ['-fopenmp', '-msse4.1', '-Wall', '-g']
    )
]

if torch.cuda.is_available() and CUDA_HOME is not None:
    extension = CUDAExtension(
        name = 'mixnet._cuda',
        include_dirs = ['./mix_src'],
        sources = [
            'mix_src/mixnet.cpp',
            'mix_src/mixnet_cuda.cu',
        ],
        extra_compile_args = {
            'cxx': ['-DMIX_USE_GPU', '-g'],
            'nvcc': ['-g', '-restrict', '-maxrregcount', '32', '-lineinfo', '-Xptxas=-v']
        }
    )
    ext_modules.append(extension)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Python interface
setup(
    name='mixnet',
    version='0.1.1',
    install_requires=['torch>=1.3'],
    packages=['mixnet'],
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    url='https://github.com/ML-PL/satnet-rules',
    zip_safe=False,
    description='Bridging deep learning and logical reasoning using a differentiable satisfiability solver',
    long_description=long_description,
    long_description_content_type="text/markdown",
)
system('rm -rf build/')