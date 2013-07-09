General
------

This is a module designed to work with the ITK v4 modular system; it implements the Young & Van Vliet recursive gaussian smoothing filter.

The module should be placed in [ITK_root]/Modules/External/. You should then configure and compile ITK as normal.

If you would like to test the GPU smoothing filter, make sure to check ITK_USE_GPU during configuration and to provide a path to a valid OpenCL installation (libraries and include directories). To do this you may have to toggle advanced mode on CMake.
