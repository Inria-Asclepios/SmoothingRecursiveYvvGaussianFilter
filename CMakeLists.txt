cmake_minimum_required(VERSION 3.10.2)

project(SmoothingRecursiveYvvGaussianFilter)
enable_testing()

if(ITK_USE_GPU)
    FIND_PACKAGE(OpenCL)
    SET(_OPENCL_CPP_INCLUDE_DIRS "${OPENCL_INCLUDE_DIRS}" CACHE PATH "" FORCE)

    if (NOT "${OPENCL_INCLUDE_DIRS}" STREQUAL "")
        SET(ITK_USE_GPU ON CACHE BOOL "" FORCE)
    endif()

    SET (TRY_RUN_INCLUDES "-I${OPENCL_INCLUDE_DIRS} ")
    SET (TRY_RUN_LIBS "-DLINK_LIBRARIES:INTERNAL=${OPENCL_LIBRARIES}")

    MESSAGE(STATUS "Testing for OpenCL support.")
    MESSAGE(STATUS "Testing for double precision support on GPU.")

    TRY_RUN(
        _OPENCL_RUN_RESULT _OPENCL_COMP_RESULT
        ${CMAKE_BINARY_DIR}/CMakeTmp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/OpenCLInfo.cpp
        CMAKE_FLAGS "${TRY_RUN_LIBS}"
        COMPILE_DEFINITIONS "${TRY_RUN_INCLUDES}"
    )

    #MESSAGE(STATUS "Try-compilation returned:" "${_OPENCL_COMP_RESULT}")
    #MESSAGE(STATUS "Try-run returned:" "${_OPENCL_RUN_RESULT}")

    #The following return values are set in OpenCLInfo.cpp
    #return 0 is "all normal"
    if("${_OPENCL_RUN_RESULT}" STREQUAL "-1")
        SET(ITK_USE_GPU "OFF" CACHE BOOL "")
        MESSAGE(STATUS "No support for OpenCL found.")
    else()
        if("${_OPENCL_RUN_RESULT}" STREQUAL "1")
            SET(GPU_HANDLES_DOUBLE "ON" CACHE BOOL "")
            add_definitions(-DWITH_DOUBLE)
            MESSAGE(STATUS "Valid OpenCL device found.")
            MESSAGE(STATUS "All devices accept double precision.")
        else()
            #if( ("${_OPENCL_RUN_RESULT}" STREQUAL "20") OR ("${_OPENCL_RUN_RESULT}" STREQUAL "10") )
            SET(GPU_HANDLES_DOUBLE "OFF" CACHE BOOL "")
            MESSAGE(STATUS "Valid OpenCL device found.")
            MESSAGE(STATUS "Setting single-precision on GPU.")
            #endif()
        endif()
    endif()
endif()

MESSAGE(STATUS "Using OpenCL: ${ITK_USE_GPU}")
MESSAGE(STATUS "GPU double precision: ${GPU_HANDLES_DOUBLE}")

#ITK_USE_GPU could have been toggled off during the test run.
if (ITK_USE_GPU)
  set(${itk-module}_LIBRARIES ${itk-module})
  set(${itk-module}_SYSTEM_INCLUDE_DIRS ${OPENCL_INCLUDE_DIRS})
  set(${itk-module}_SYSTEM_LIBRARY_DIRS ${OPENCL_LIBRARIES})
  set(${itk-module}KernelDir "${CMAKE_CURRENT_BINARY_DIR}/src" CACHE PATH "" FORCE)

  option(NVIDIA_GPU "Check if your GPU is NVidia (affects optimizations). Default: true." TRUE)
else()
    #No GPU, but CPU user will probably want double-precision.
    add_definitions(-DWITH_DOUBLE)
endif()
itk_module_impl()
