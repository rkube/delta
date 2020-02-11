
# python setup.py build_ext --inplace


from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy 


extensions = [
    Extension("diagnostics_cython", ["diagnostics_cython.pyx"],
        include_dirs=[numpy.get_include()]),
]

setup(name="diag_cython", ext_modules=cythonize(extensions))