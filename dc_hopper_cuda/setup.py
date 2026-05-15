from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="_dc_hopper_cuda",
    ext_modules=[
        CUDAExtension(
            "_dc_hopper_cuda",
            ["dc_hopper_api.cpp", "dc_hopper_kernel.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-lineinfo",
                    "--threads=4",
                    "-gencode=arch=compute_90a,code=sm_90a",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
