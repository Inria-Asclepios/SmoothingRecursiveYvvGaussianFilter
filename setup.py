# -*- coding: utf-8 -*-
from __future__ import print_function
from os import sys

try:
    from skbuild import setup
except ImportError:
    print('scikit-build is required to build from source.', file=sys.stderr)
    print('Please run:', file=sys.stderr)
    print('', file=sys.stderr)
    print('  python -m pip install scikit-build')
    sys.exit(1)

setup(
    name='smoothingrecursiveyvvgaussianfilter',
    version='0.0.1',
    author='Irina Vidal Migallón',
    author_email='irina.vidal-migallon@inria.fr',
    packages=['itk'],
    package_dir={'itk': 'itk'},
    download_url=r'https://github.com/InsightSoftwareConsortium/SmoothingRecursiveYvvGaussianFilter',
    description=r'Young & Van Vliet recursive Gaussian smoothing filter for GPU (OpenCL) and CPU',
    long_description='SmoothingRecursiveYvvGaussianFilter provides a GPU'
                     '(OpenCL) and CPU implementation of the computationally'
                     'efficient forward and backward IIR Young & Van Vliet'
                     'recursive Gaussian smoothing filter.\n'
                     'Please refer to:'
                     'Vidal-Migallon I., Commowick O., Pennec X., Dauguet J., Vercauteren T.,'
                     '“GPU and CPU implementation of Young - Van Vliet\'s Recursive Gaussian Smoothing Filter”, '
                     '“Fostering Open Science in Lung Cancer Lesion Sizing with ITK module LSTK”, '
                     'Insight Journal, http://hdl.handle.net/10380/3425, 2013.',
    classifiers=[
        "License :: OSI Approved :: BSD 2-Clause \"Simplified\" License",
        "Programming Language :: Python",
        "Programming Language :: C++",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries",
        "Operating System :: Android",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS"
        ],
    license='BSD',
    keywords='ITK InsightToolkit',
    url=r'https://itk.org/',
    install_requires=[
        r'itk'
    ]
    )
