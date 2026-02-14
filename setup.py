from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "APDNN",
        ["bindings.cpp", 
         "src/tensor.cpp", 
         "src/ops.cpp",
         "src/linear.cpp",
         "src/init.cpp",
         "src/optimizer.cpp",
         "src/loss.cpp"],
        include_dirs=["include"],
        cxx_std=17,
    ),
]

setup(
    name="APDNN",
    version="0.1.0",
    author="Avinash, Pranay",
    description="C++ CNN Backend with Python bindings",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)