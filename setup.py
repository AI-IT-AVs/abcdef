from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='posdiff',
    version='1.0.0',
    ext_modules=[
        CUDAExtension(
            name='posdiff.ext',
            sources=[
                'posdiff/extensions/extra/cloud/cloud.cpp',
                'posdiff/extensions/cpu/grid_subsampling/grid_subsampling.cpp',
                'posdiff/extensions/cpu/grid_subsampling/grid_subsampling_cpu.cpp',
                'posdiff/extensions/cpu/radius_neighbors/radius_neighbors.cpp',
                'posdiff/extensions/cpu/radius_neighbors/radius_neighbors_cpu.cpp',
                'posdiff/extensions/pybind.cpp',
            ],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
