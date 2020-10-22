
# CC=icc  LDSHARED="icc -shared" python setup.py build_ext --inplace


from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy 

extensions = [
    Extension("diagnostics_cython", ["diagnostics_cython.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-qopenmp',  '-qopt-report=5'],
        extra_link_args=['-qopenmp']),
]

setup(name="diag_cython", ext_modules=cythonize(extensions, compiler_directives={'language_level' : "3"}))