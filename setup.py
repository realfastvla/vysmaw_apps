from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import sys
sys.path.append('/home/cbe-master/wcbe/py/lib/python2.7/site-packages/vysmaw')

extensions = [
    Extension("*", ["*.pyx"],
        libraries = ["vysmaw", "vys", "python2.7"],
        include_dirs = ["/opt/cbe-local/include"],
        library_dirs = ["/home/cbe-master/wcbe/py/lib/python2.7/site-packages/vysmaw", "/opt/cbe-local/lib"],)]

setup(
  name = 'vysmaw applications',
  ext_modules = cythonize(extensions),
)
