project(ITKSmoothingRecursiveYvvGaussianFilter)
enable_testing()

SET (TRY_RUN_INCLUDES "-I${OPENCL_INCLUDE} ")
SET (TRY_RUN_LIBS "-DLINK_LIBRARIES:INTERNAL=${OPENCL_LIBRARIES}")
TRY_RUN(
    _OPENCL_VERSION_RUN_RESULT _OPENCL_VERSION_RUN_RESULT
    ${CMAKE_BINARY_DIR}/CMakeTmp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/OpenCLInfo.cpp
    CMAKE_FLAGS "${TRY_RUN_LIBS}"
    COMPILE_DEFINITIONS "${TRY_RUN_INCLUDES}"
)
MESSAGE(STATUS "------------try run result errors: ${_OPENCL_VERSION_RUN_RESULT}")

#The following return values are set in OpenCLInfo.cpp
#return 0 is "all normal"
if("${_OPENCL_VERSION_RUN_RESULT}" STREQUAL "0") 
    SET(GPU_HANDLES_DOUBLE "ON" CACHE BOOL "")
    add_definitions(-DWITH_DOUBLE)
    MESSAGE(STATUS "All devices accept double precision.")
endif()

if("${_OPENCL_VERSION_RUN_RESULT}" STREQUAL "100") 
    SET(ITK_USE_GPU "OFF" CACHE BOOL "")
    MESSAGE(STATUS "No opencl-capable devices found.")
else()
    SET(ITK_USE_GPU "ON" CACHE BOOL "")
endif()

if (NOT "${_OPENCL_CPP_INCLUDE_DIRS}" STREQUAL "")
    SET(WITH_OPENCL ON CACHE BOOL "" FORCE) 
endif()

MESSAGE(STATUS "Using OpenCL: ${WITH_OPENCL}")       

if (ITK_USE_GPU AND WITH_OPENCL)

  set(ITKSmoothingRecursiveYvvGaussianFilter_LIBRARIES ITKSmoothingRecursiveYvvGaussianFilter)
  set(ITKSmoothingRecursiveYvvGaussianFilter_SYSTEM_INCLUDE_DIRS ${OPENCL_INCLUDE_DIRS})
  set(ITKSmoothingRecursiveYvvGaussianFilter_SYSTEM_LIBRARY_DIRS ${OPENCL_LIBRARIES})
  set(ITKSmoothingRecursiveYvvGaussianFilterKernelDir "${CMAKE_CURRENT_BINARY_DIR}/src" CACHE PATH "" FORCE)

  option(NVIDIA_GPU "Uncheck if your GPU is not NVidia (affects optimizations)." TRUE)
    #option(GPU_HANDLES_DOUBLE "GPU handles double precision (i.e. calculation capability 1.3 or better)." FALSE)

endif()

#set(TEST_DATA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/Data" CACHE PATH "")
itk_module_impl()

#add_subdirectory(Example)
