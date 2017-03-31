from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import sys
sys.path.append('/opt/cbe-local/lib/python2.7/site-packages/vysmaw')


extensions = [
    Extension("*", ["*.pyx"],
        libraries = ["vysmaw", "vys", "python2.7"],
        include_dirs = ["/opt/cbe-local/include"],
        library_dirs = ["/opt/cbe-local/lib"],) #, "/opt/cbe-local/lib/python2.7/site-packages/vysmaw/"],)
#        include_dirs = ["/home/cbe-master/realfast/anaconda/include", "/home/cbe-master/realfast/anaconda/include/python2.7"],
#        library_dirs = ["/home/cbe-master/realfast/anaconda/lib/python2.7/site-packages"],)
]

setup(
  name = 'vysmaw applications',
  ext_modules = cythonize(extensions),
)
