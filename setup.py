#from distutils.core import setup
#from distutils.extension import Extension
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import sys
#sys.path.append('/home/cbe-master/wcbe/py/lib/python2.7/site-packages/vysmaw')
sys.path.append("/opt/cbe-local/lib/python3.6/site-packages/vysmaw")

extensions = [
    Extension("*", ["*.pyx"],
        libraries = ["vysmaw", "vys", "python3.6m"],
        include_dirs = ["/opt/cbe-local/include"],
#        library_dirs = ["/home/cbe-master/wcbe/py/lib/python2.7/site-packages/vysmaw", "/opt/cbe-local/lib"],)]
        library_dirs = ["/opt/cbe-local/lib/python3.6/site-packages/vysmaw", "/opt/cbe-local/lib"],)]

setup(
  name = 'vysmaw applications',
  packages = find_packages(),
  include_package_data = True,
  cmdclass = {'build_ext': build_ext},
  ext_modules = cythonize(extensions),
  py_modules=['vysmawcli'],
  install_requires=['Click'],
  entry_points='''
  [console_scripts]
  vyscheck=vysmawcli:vyscheck
  ''',
)
