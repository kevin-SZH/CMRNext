"""点云可见性 CUDA 扩展构建。

本模块属于 CMRNext 项目自有源码；函数和类的参数、返回值、异常副作用以及
涉及的坐标系、张量布局和单位均在对应中文 docstring 中说明。
"""

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cxx_args = ['-std=c++17'] # Use C++17 for current PyTorch.

nvcc_args = [
    '-gencode', 'arch=compute_60,code=sm_60',
    '-gencode', 'arch=compute_70,code=sm_70',
    '-gencode', 'arch=compute_80,code=sm_80',
    '-gencode', 'arch=compute_86,code=sm_86',
    # sm_89 and sm_90 require CUDA 12+.
    # CUDA 11.3 uses sm_86 PTX JIT compatibility on newer GPUs.
    # '-gencode', 'arch=compute_89,code=sm_89',
    # '-gencode', 'arch=compute_90,code=sm_90',
]

setup(
    name='visibility',
    version='0.1',
    author="Daniele Cattaneo",
    author_email="cattaneo@cs.uni-freiburg.de",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension('visibility', [
            'src/visibility.cpp',
            'src/visibility_kernel.cu',
        ], extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args})
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
)
