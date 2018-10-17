SmoothingRecursiveYvvGaussianFilter
===================================

.. |CircleCI| image:: https://circleci.com/gh/InsightSoftwareConsortium/SmoothingRecursiveYvvGaussianFilter.svg?style=shield
    :target: https://circleci.com/gh/InsightSoftwareConsortium/SmoothingRecursiveYvvGaussianFilter

.. |TravisCI| image:: https://travis-ci.org/InsightSoftwareConsortium/SmoothingRecursiveYvvGaussianFilter.svg?branch=master
    :target: https://travis-ci.org/InsightSoftwareConsortium/SmoothingRecursiveYvvGaussianFilter

.. |AppVeyor| image:: https://img.shields.io/appveyor/ci/InsightSoftwareConsortium/smoothingrecursiveyvvgaussianfilter.svg
    :target: https://ci.appveyor.com/project/InsightSoftwareConsortium/smoothingrecursiveyvvgaussianfilter

=========== =========== ===========
   Linux      macOS       Windows
=========== =========== ===========
|CircleCI|  |TravisCI|  |AppVeyor|
=========== =========== ===========


Overview
--------

This is a module for the `Insight Toolkit (ITK) <http://itk.org>`_ that
implements the Young & Van Vliet recursive Gaussian smoothing filter for GPU
(OpenCL) and CPU.

For more information, see the `Insight Journal article <http://hdl.handle.net/10380/3425>`_::

  Vidal-Migallon I., Commowick O., Pennec X., Dauguet J., Vercauteren T.
  GPU and CPU implementation of Young - Van Vliet's Recursive Gaussian Smoothing Filter
  The Insight Journal. January-December. 2013.
  http://hdl.handle.net/10380/3425
  http://www.insight-journal.org/browse/publication/896


Installation
------------

If you would like to test the GPU smoothing filter, make sure to check the
flag `ITK_USE_GPU` during configuration and to provide a path to a valid
OpenCL installation (libraries and include directories). To do this you may
have to toggle advanced mode on CMake.


License
-------

This software is distributed under the Apache 2.0 license. Please see
the *LICENSE* file for details.
