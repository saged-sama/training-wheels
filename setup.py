from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "therapml_cpp",
        ["therapml/bindings.cpp", "therapml/matrix/matrix_ops.cpp"],
        include_dirs=["therapml/matrix"],
        cxx_std=11 # Or 17 if you use newer features
    ),
]

setup(name="therapml", ext_modules=ext_modules, cmdclass={"build_ext": build_ext})