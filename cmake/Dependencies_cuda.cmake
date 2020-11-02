# Defines the following output variables:

unset(PACKAGE_INCLUDES)
unset(LIB_TARGETS)
unset(LIBS)
unset(MODULE_CUDA)

unset(CUDA_FOUND)
unset(CMAKE_CUDA_COMPILER)


find_package(CUDAToolkit 10.2)

#Check Language is an extra module so we need to include it.
include(CheckLanguage)
check_language(CUDA)

set(SAIGA_USE_CUDA_TOOLKIT 0)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  message(STATUS "Enabled CUDA. Version: ${CUDAToolkit_VERSION}" )
  set(CUDA_FOUND TRUE)
  set(SAIGA_USE_CUDA 1)
  set(MODULE_CUDA 1)
  if(CUDAToolkit_FOUND)
    set(SAIGA_USE_CUDA_TOOLKIT 1)
    endif()
else()
  message(STATUS "CUDA not found.")
  set(CUDA_FOUND FALSE)
  SET(SAIGA_USE_CUDA 0)
  set(MODULE_CUDA 0)
endif()


if(CUDA_FOUND)
  # Cuda Runtime
  PackageHelperTarget(CUDA::cudart CUDA_FOUND)

  # Image processing
  PackageHelperTarget(CUDA::nppif CUDA_FOUND)
  PackageHelperTarget(CUDA::nppig CUDA_FOUND)

  # Debuging tools
  PackageHelperTarget(CUDA::nvToolsExt CUDA_FOUND)


  if(SAIGA_CUDA_RDC)
    # for dynamic parallelism
    list(APPEND SAIGA_CUDA_FLAGS "--relocatable-device-code=true")
  endif()


  set(SAIGA_CUDA_ARCH "52-virtual")

  if(NOT MSVC)
    list(APPEND SAIGA_CUDA_FLAGS "-Xcompiler=-fopenmp")
  else()
    list(APPEND SAIGA_CUDA_FLAGS "-Xcompiler=/openmp")
  endif()

  list(APPEND SAIGA_CUDA_FLAGS "-use_fast_math")
  list(APPEND SAIGA_CUDA_FLAGS "--expt-relaxed-constexpr")
  list(APPEND SAIGA_CUDA_FLAGS "-Xcudafe=--diag_suppress=esa_on_defaulted_function_ignored")

  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND ${SAIGA_LIBSTDCPP})
    list(APPEND SAIGA_CUDA_FLAGS "-Xcompiler=-stdlib=libstdc++")
  endif()

  if(BUILD_SHARED)
    list(APPEND SAIGA_CUDA_FLAGS "-Xcompiler=-DSAIGA_DLL_EXPORTS")
  endif()

  set(CMAKE_CUDA_STANDARD 14)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()


set(CUDA_INCLUDES ${PACKAGE_INCLUDES})
set(CUDA_LIBS ${LIBS})
set(CUDA_TARGETS saiga_core saiga_vision ${LIB_TARGETS})

