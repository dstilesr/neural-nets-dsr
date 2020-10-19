from setuptools import Extension, setup, find_packages
import os
import numpy as np
from Cython.Build import cythonize

ext = [Extension(
    "neural_nets_dsr.layers.numeric_utils.conv_utils",
    [os.path.join(
        "neural_nets_dsr",
        "layers",
        "numeric_utils",
        "conv_utils.pyx"
    )],
    extra_compile_args=["-fopenmp"],
    extra_link_args=["-fopenmp"],
    include_dirs=[np.get_include()]
)]

setup(
    name="neural_nets_dsr",
    version="0.0.1",
    author="David Stiles Rosselli",
    url="https://github.com/dstilesr/neural-nets-dsr",
    ext_modules=cythonize(ext, include_path=[np.get_include()]),
    packages=find_packages(),
    install_requires=[
        "Cython",
        "numpy"
    ],
    zip_safe=False
)
