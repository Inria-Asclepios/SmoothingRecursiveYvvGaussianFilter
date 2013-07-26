General
------

This is a module designed to work with the ITK v4 modular system; it implements the Young & Van Vliet recursive gaussian smoothing filter for GPU (OpenCL) and CPU. For details on the implementation, please refer to the Insight Journal publication: http://hdl.handle.net/10380/3425

Build
-----

The preferred way to build this module is activating the option Fetch_SmoothingRecursiveYvvGaussianFilter during CMake configuration of ITK. The module will be downloaded and compiled with all other ITK modules.

A secondary way is downloading the latest version from this repository and placing the module in [ITK_root]/Modules/External/. You should then configure and compile ITK as normal.

If you would like to test the GPU smoothing filter, make sure to check ITK_USE_GPU during configuration and to provide a path to a valid OpenCL installation (libraries and include directories). To do this you may have to toggle advanced mode on CMake.
