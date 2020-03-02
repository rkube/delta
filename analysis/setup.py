# CC=icc  LDSHARED="icc -shared" python setup.py build_ext --inplace


from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy 

extensions = [
    Extension("kernels_spectral_cy", 
              sources=["kernels_spectral_cy.pyx"],
              libraries=["kernels"],
              library_dirs=["lib"],
              include_dirs=[numpy.get_include(), "lib"],
              extra_compile_args=['-qopenmp'],
              extra_link_args=['-qopenmp']),
]

setup(name="kernels_spectral_cy", 
      ext_modules=cythonize(extensions, compiler_directives={'language_level' : "3"}))