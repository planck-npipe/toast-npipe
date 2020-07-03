#!/usr/bin/env python

# This assumes that it will be run with compilers set to the
# Intel compilers:
#
# $> CC=icc CXX=icpc LDSHARED="icc -shared" python setup.py ...
#

import glob
import os
import re
import unittest

from Cython.Build import cythonize
from setuptools import setup, Extension
from setuptools.command.test import test as TestCommand

import numpy as np


def get_version():
    ver = 'unknown'
    if os.path.isfile("toast_planck/_version.py"):
        f = open("toast_planck/_version.py", "r")
        for line in f.readlines():
            mo = re.match("__version__ = '(.*)'", line)
            if mo:
                ver = mo.group(1)
        f.close()
    return ver


current_version = get_version()

# extensions to build

ext_signal_estimator = Extension(
    'toast_planck.preproc_modules.signal_estimation',
    sources=['toast_planck/preproc_modules/signal_estimation.pyx',
             'toast_planck/preproc_modules/despike/medianmap.c'],
    include_dirs=[np.get_include(),
                  'toast_planck/preproc_modules/despike/include'],
    libraries=['imf'],
)

ext_zodier = Extension(
    'toast_planck.reproc_modules.zodi',
    sources=['toast_planck/reproc_modules/zodi.pyx'],
    include_dirs=[np.get_include()],
    libraries=['imf'],
)

ext_time_response_tools = Extension(
    'toast_planck.preproc_modules.time_response_tools',
    include_dirs=[np.get_include()],
    sources=['toast_planck/preproc_modules/time_response_tools.pyx'],
)

ext_destripe_tools = Extension(
    'toast_planck.reproc_modules.destripe_tools',
    include_dirs=[np.get_include()],
    sources=['toast_planck/reproc_modules/destripe_tools.pyx'],
)

ext_despyke = Extension(
    'toast_planck.preproc_modules.despyke',
    sources=['toast_planck/preproc_modules/despike/despyke.pyx',
             'toast_planck/preproc_modules/despike/despike_func.cc',
             'toast_planck/preproc_modules/despike/medianmap.c',
             'toast_planck/preproc_modules/despike/ring_spline2d.c',
             'toast_planck/preproc_modules/despike/spline.c',
             'toast_planck/preproc_modules/despike/spline2d.c',
             'toast_planck/preproc_modules/despike/todprocess_planck.cc'],
    include_dirs=[np.get_include(),
                  'toast_planck/preproc_modules/despike/include'],
    libraries=['fftw3', 'fftw3_threads', 'mkl_rt', 'iomp5', 'pthread', 'imf',
               'svml'],
    library_dirs=[],
    language='c++',
    extra_compile_args=['-qopenmp'],
    extra_link_args=['-qopenmp'],
)

ext_shdet = Extension(
    'toast_planck.shdet',
    sources=['toast_planck/shdet/shdet.pyx',
             'toast_planck/shdet/shdet_func.cpp',
             'toast_planck/shdet/cshdet.cpp'],
    include_dirs=[np.get_include(),
                  'toast_planck/shdet/include'],
    language='c++',
)

extensions = cythonize([
    ext_signal_estimator, ext_zodier, ext_time_response_tools, ext_despyke,
    ext_destripe_tools, ext_shdet])

# scripts to install

scripts = glob.glob('pipelines/*.py')

# run unit tests


class PTestCommand(TestCommand):

    def __init__(self, *args, **kwargs):
        super(PTestCommand, self).__init__(*args, **kwargs)

    def initialize_options(self):
        TestCommand.initialize_options(self)

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_suite = True

    def run(self):
        loader = unittest.TestLoader()
        runner = unittest.TextTestRunner(verbosity=2)
        suite = loader.discover('tests', pattern='test_*.py', top_level_dir='.')
        runner.run(suite)

# set it all up


setup(
    name='toast_planck',
    provides='toast_planck',
    version=current_version,
    description='Planck extensions to TOAST',
    author='Brendan Crill, Sergi Hildebrandt, Reijo Keskitalo, Ted Kisner, '
    'Guillaume Patanchon, Cyrille Rosset, Gael Roudier',
    author_email='mail@theodorekisner.com',
    url='https://github.com/tskisner/toast-planck',
    packages=['toast_planck', 'toast_planck.preproc_modules',
              'toast_planck.reproc_modules', 'toast_planck.beam_modules'],
    package_data={'toast_planck.preproc_modules': [
        'ephemeris/*txt', 'lfi_adc_data/*pic', 'lfi_fsl_data/*csv']},
    ext_modules=extensions,
    scripts=scripts,
    license='BSD',
    requires=['Python (>3.4.0)', ],
    cmdclass={'test': PTestCommand}
)
