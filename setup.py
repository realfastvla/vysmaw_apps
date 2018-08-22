#from distutils.core import setup
#from distutils.extension import Extension
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

extensions = [
    Extension("*", ["*.pyx"],
        libraries = ["vysmaw", "vys", "python3.6m"],
#        include_dirs = ["/opt/cbe-local/include", "."],
#        library_dirs = ["/home/cbe-master/realfast/anaconda/envs/development3/lib/python3.6/site-packages", "/opt/cbe-local/lib"],
        include_dirs = ["/users/mpokorny/projects/vysmaw/src", "."],
        library_dirs = ["/home/cbe-master/realfast/anaconda/envs/development3/lib/python3.6/site-packages", "/users/mpokorny/projects/vysmaw/build/src"],
        extra_compile_args = ["-fno-strict-aliasing"])]


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
