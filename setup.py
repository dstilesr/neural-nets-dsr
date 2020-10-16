from setuptools import setup
from setuptools import find_packages, Extension
import os
import numpy as np
from Cython.Build import cythonize


setup(
    name="neural_nets_dsr",
    version="0.0.1",
    author="David Stiles Rosselli",
    url="https://github.com/dstilesr/neural-nets-dsr",
    # ext_modules=[cythonize(
    #     os.path.join(
    #         "neural_nets_dsr",
    #         "layers",
    #         "conv_utils",
    #         "conv_numeric.pyx"
    #     ),
    #     include_path=[np.get_include()]
    # )],
    ext_modules=[Extension(
        "neural_nets_dsr.layers.conv_utils.conv_numeric",
        ["neural_nets_dsr/layers/conv_utils/conv_numeric.c"],
        include_dirs=[np.get_include()]
    )],
    packages=find_packages(),
    install_requires=[
        "Cython",
        "numpy"
    ],
    # include_dirs=np.get_include(),
    zip_safe=False
)
