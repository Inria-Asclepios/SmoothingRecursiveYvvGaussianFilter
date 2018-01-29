General
------

This is a module designed to work with the ITK modular system; it implements the Young & Van Vliet recursive gaussian smoothing filter for GPU (OpenCL) and CPU. For details on the implementation, please refer to the Insight Journal publication: http://hdl.handle.net/10380/3425

Build
-----

Option 1: The preferred way to build this module is activating the option Fetch_SmoothingRecursiveYvvGaussianFilter during CMake configuration of ITK. The module will be downloaded (to [ITK_root]/Modules/Remote/) and compiled with all other ITK modules.

Option 2: A secondary way is downloading the latest version from this repository and placing the module in [ITK_root]/Modules/External/. You should then configure and compile ITK as normal.

If you would like to test the GPU smoothing filter, make sure to check ITK_USE_GPU during configuration and to provide a path to a valid OpenCL installation (libraries and include directories). To do this you may have to toggle advanced mode on CMake.

License and copyright
-----

Copyright Insight Software Consortium

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0.txt

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

