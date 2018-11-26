cmake_minimum_required(VERSION 3.10.2)
project(SmoothingRecursiveYvvGaussianFilter)

enable_testing()

if(ITK_USE_GPU)
  find_package(OpenCL)
  set(_OPENCL_CPP_INCLUDE_DIRS "${OPENCL_INCLUDE_DIRS}" CACHE PATH "" FORCE)

  if(NOT "${OPENCL_INCLUDE_DIRS}" STREQUAL "")
    set(ITK_USE_GPU ON CACHE BOOL "" FORCE)
  endif()

  set(TRY_RUN_INCLUDES "-I${OPENCL_INCLUDE_DIRS} ")
  set(TRY_RUN_LIBS "-DLINK_LIBRARIES:INTERNAL=${OPENCL_LIBRARIES}")

  message(STATUS "Testing for OpenCL support.")
  message(STATUS "Testing for double precision support on GPU.")

  try_run(
    _OPENCL_RUN_RESULT _OPENCL_COMP_RESULT
    ${CMAKE_BINARY_DIR}/CMakeTmp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/OpenCLInfo.cpp
    CMAKE_FLAGS "${TRY_RUN_LIBS}"
    COMPILE_DEFINITIONS "${TRY_RUN_INCLUDES}"
  )

  #message(STATUS "Try-compilation returned:" "${_OPENCL_COMP_RESULT}")
  #message(STATUS "Try-run returned:" "${_OPENCL_RUN_RESULT}")

  #The following return values are set in OpenCLInfo.cpp
  #return 0 is "all normal"
  if("${_OPENCL_RUN_RESULT}" STREQUAL "-1")
    set(ITK_USE_GPU "OFF" CACHE BOOL "")
    message(STATUS "No support for OpenCL found.")
  else()
    if("${_OPENCL_RUN_RESULT}" STREQUAL "1")
      set(GPU_HANDLES_DOUBLE "ON" CACHE BOOL "")
      add_definitions(-DWITH_DOUBLE)
      message(STATUS "Valid OpenCL device found.")
      message(STATUS "All devices accept double precision.")
    else()
      #if(("${_OPENCL_RUN_RESULT}" STREQUAL "20") OR ("${_OPENCL_RUN_RESULT}" STREQUAL "10"))
      set(GPU_HANDLES_DOUBLE "OFF" CACHE BOOL "")
      message(STATUS "Valid OpenCL device found.")
      message(STATUS "Setting single-precision on GPU.")
      #endif()
    endif()
  endif()
endif()

message(STATUS "Using OpenCL: ${ITK_USE_GPU}")
message(STATUS "GPU double precision: ${GPU_HANDLES_DOUBLE}")

#ITK_USE_GPU could have been toggled off during the test run.
if(ITK_USE_GPU)
  set(${itk-module}_LIBRARIES ${itk-module})
  set(${itk-module}_SYSTEM_INCLUDE_DIRS ${OPENCL_INCLUDE_DIRS})
  set(${itk-module}_SYSTEM_LIBRARY_DIRS ${OPENCL_LIBRARIES})
  set(${itk-module}KernelDir "${CMAKE_CURRENT_BINARY_DIR}/src" CACHE PATH "" FORCE)

  option(NVIDIA_GPU "Check if your GPU is NVidia (affects optimizations). Default: true." TRUE)
else()
  #No GPU, but CPU user will probably want double-precision.
  add_definitions(-DWITH_DOUBLE)
endif()


if(NOT ITK_SOURCE_DIR)
  find_package(ITK REQUIRED)
  list(APPEND CMAKE_MODULE_PATH ${ITK_CMAKE_DIR})
  include(ITKModuleExternal)
else()
  itk_module_impl()
endif()
