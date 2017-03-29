from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension("*", ["*.pyx"],
        include_dirs = ["/home/cbe-master/realfast/anaconda/include", "/home/cbe-master/realfast/anaconda/include/python2.7"],
        libraries = ["vysmaw", "vys", "python2.7"],
        library_dirs = ["/home/cbe-master/realfast/anaconda/lib/python2.7/site-packages"],)
]

setup(
  name = 'vysmaw applications',
  ext_modules = cythonize(extensions),
)
