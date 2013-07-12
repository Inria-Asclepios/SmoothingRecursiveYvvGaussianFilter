project(ITKSmoothingRecursiveYvvGaussianFilter)
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

    MESSAGE(STATUS "Try-compilation returned:" "${_OPENCL_COMP_RESULT}")
    MESSAGE(STATUS "Try-run returned:" "${_OPENCL_RUN_RESULT}")

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
        endif()        
    endif()
endif()

MESSAGE(STATUS "Using OpenCL: ${ITK_USE_GPU}")       

#ITK_USE_GPU could have been toggled off during the test run.
if (ITK_USE_GPU)# AND OPENCL_FOUND)
  set(ITKSmoothingRecursiveYvvGaussianFilter_LIBRARIES ITKSmoothingRecursiveYvvGaussianFilter)
  set(ITKSmoothingRecursiveYvvGaussianFilter_SYSTEM_INCLUDE_DIRS ${OPENCL_INCLUDE_DIRS})
  set(ITKSmoothingRecursiveYvvGaussianFilter_SYSTEM_LIBRARY_DIRS ${OPENCL_LIBRARIES})
  set(ITKSmoothingRecursiveYvvGaussianFilterKernelDir "${CMAKE_CURRENT_BINARY_DIR}/src" CACHE PATH "" FORCE)

  option(NVIDIA_GPU "Check if your GPU is NVidia (affects optimizations). Default: true." TRUE)
  #option(GPU_HANDLES_DOUBLE "GPU handles double precision (i.e. calculation capability 1.3 or better)." FALSE)

endif()

#set(TEST_DATA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/Data" CACHE PATH "")
itk_module_impl()

#add_subdirectory(Example)
