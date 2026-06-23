from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="fused_attention",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="fused_attention._C",
            sources=["csrc/bindings.cpp", "csrc/attention.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-gencode", "arch=compute_80,code=sm_80"],
            },
            
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
