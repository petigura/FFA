from setuptools import setup
from setuptools.extension import Extension
from Cython.Distutils import build_ext
# from Cython.Build import cythonize
import numpy

ext_modules = [
                Extension(name="FFA_cy",
                         sources=["FFA/FFA_cy.pyx"],
                         include_dirs=[numpy.get_include()]),
               Extension(name="FBLS_cy",
                         sources=["FFA/FBLS_cy.pyx"],
                         include_dirs=[numpy.get_include()]),
               Extension(name="BLS_cy",
                         sources=["FFA/BLS_cy.pyx"],
                         include_dirs=[numpy.get_include()]),
               Extension(name="FFA_cext",
                         sources=["FFA/FFA_cext.pyx", "FFA/FFA.c"],
                         include_dirs=[numpy.get_include()]),
               Extension(name="FBLS_cext",
                         sources=["FFA/FBLS_cext.pyx", "FFA/FBLS.c"],
                         include_dirs=[numpy.get_include()]),
               Extension(name="BLS_cext",
                         sources=["FFA/BLS_cext.pyx", "FFA/BLS.c"],
                         include_dirs=[numpy.get_include()]),
               Extension(name="fold",
                         sources=["FFA/fold.pyx"],
                         include_dirs=[numpy.get_include()])
               ]

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
    name='FFA',
    packages=['FFA'],
    package_data = {
        'FFA': ['sample_data/*']
    },
    requires=['numpy', 'cython'],
    version='0.0.2',
)
