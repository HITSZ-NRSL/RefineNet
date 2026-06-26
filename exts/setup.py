from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='BpOps_and_GnnOps',   # 名字无所谓，但推荐改成更通用的
    ext_modules=[
        # 原有 BpOps
        CUDAExtension(
            'BpOps',
            [
                'bp_cuda.cpp',
                'bp_cuda_kernel.cu',
            ],
            extra_compile_args={
                'cxx':  ['-g'],
                'nvcc': ['-O3']
            }
        ),
        CUDAExtension(
            'gnn_knn',
            [
                'knn_bind.cpp',        # 你创建的绑定文件
                'knn_kernel.cu',       # 你创建的 CUDA KNN kernel
            ],
            extra_compile_args={
                'cxx':  ['-g'],
                'nvcc': ['-O3']
            }
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)
