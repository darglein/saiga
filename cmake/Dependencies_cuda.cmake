# Defines the following output variables:

unset(PACKAGE_INCLUDES)
unset(LIB_TARGETS)
unset(LIBS)
unset(MODULE_CUDA)

unset(CUDA_FOUND)
unset(CMAKE_CUDA_COMPILER)


message(STATUS "SAIGA_CUDA_VERSION ${SAIGA_CUDA_VERSION}")
find_package(CUDAToolkit 10.2)


#Check Language is an extra module so we need to include it.
#include(CheckLanguage)
#check_language(CUDA)

if (CUDAToolkit_FOUND)
    enable_language(CUDA)
    message(STATUS "Enabled CUDA. Version: ${CUDAToolkit_VERSION}")
    set(CUDA_FOUND TRUE)
    set(SAIGA_USE_CUDA 1)
    set(MODULE_CUDA 1)
    set(SAIGA_USE_CUDA_TOOLKIT 1)
else ()
    message(STATUS "CUDA not found.")
    set(CUDA_FOUND FALSE)
    SET(SAIGA_USE_CUDA 0)
    set(MODULE_CUDA 0)
endif ()


if (CUDA_FOUND)
    # Cuda Runtime
    PackageHelperTarget(CUDA::cudart CUDA_FOUND)

    # Image processing
    PackageHelperTarget(CUDA::nppif CUDA_FOUND)
    PackageHelperTarget(CUDA::nppig CUDA_FOUND)

    # Debuging tools
    PackageHelperTarget(CUDA::nvToolsExt CUDA_FOUND)


    if (SAIGA_CUDA_RDC)
        # for dynamic parallelism
        list(APPEND SAIGA_CUDA_FLAGS "--relocatable-device-code=true")
    endif ()

    if (CUDAToolkit_VERSION)
        set(SAIGA_CUDA_VERSION ${CUDAToolkit_VERSION} CACHE STRING "Detected CUDA Version")
    elseif (CUDA_VERSION)
        set(SAIGA_CUDA_VERSION ${CUDA_VERSION} CACHE STRING "Detected CUDA Version")
    elseif (SAIGA_CUDA_VERSION)
    else ()
        message(FATAL_ERROR "Unknown cuda version ${SAIGA_CUDA_VERSION} ${CUDAToolkit_VERSION} ${CUDA_VERSION}")
    endif ()


    if (NOT MSVC)
        list(APPEND SAIGA_CUDA_FLAGS "-Xcompiler=-fopenmp")

        if (SAIGA_FULL_OPTIMIZE OR SAIGA_ARCHNATIVE)
            list(APPEND SAIGA_CUDA_FLAGS "-Xcompiler=-march=native")
        endif ()
    else ()
        list(APPEND SAIGA_CUDA_FLAGS "-Xcompiler=/openmp")
        list(APPEND SAIGA_CUDA_FLAGS "-Xcompiler=/W0")
    endif ()

    list(APPEND SAIGA_CUDA_FLAGS "-use_fast_math")
    list(APPEND SAIGA_CUDA_FLAGS "--expt-relaxed-constexpr")
    list(APPEND SAIGA_CUDA_FLAGS "-Xcudafe=--diag_suppress=esa_on_defaulted_function_ignored")

    # suppress some warnings in windows
    list(APPEND SAIGA_CUDA_FLAGS "-Xcudafe=--diag_suppress=field_without_dll_interface")
    list(APPEND SAIGA_CUDA_FLAGS "-Xcudafe=--diag_suppress=base_class_has_different_dll_interface")
    list(APPEND SAIGA_CUDA_FLAGS "-Xcudafe=--diag_suppress=dll_interface_conflict_none_assumed")
    list(APPEND SAIGA_CUDA_FLAGS "-Xcudafe=--diag_suppress=dll_interface_conflict_dllexport_assumed")

    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND ${SAIGA_LIBSTDCPP})
        list(APPEND SAIGA_CUDA_FLAGS "-Xcompiler=-stdlib=libstdc++")
    endif ()

    if (BUILD_SHARED)
        list(APPEND SAIGA_CUDA_FLAGS "-Xcompiler=-DSAIGA_DLL_EXPORTS")
    endif ()
    message(STATUS "SAIGA_CUDA_FLAGS: ${SAIGA_CUDA_FLAGS}")


    #  # 30 GTX 7xx
    #  # 52 GTX 9xx
    #  # 61 GTX 10xx
    #  # 75 RTX 20xx
    #  # 86 RTX 30xx
    #  if(${SAIGA_CUDA_VERSION} VERSION_LESS "11")
    #    set(SAIGA_CUDA_ARCH "30-virtual" "52-virtual" CACHE STRING "The cuda architecture used for compiling .cu files")
    #  else()
    #    # CUDA 11 and later doesn't support 30 anymore
    #    set(SAIGA_CUDA_ARCH "52-virtual" "75-virtual" CACHE STRING "The cuda architecture used for compiling .cu files")
    #  endif()

    include(select_compute_arch)

    if (SAIGA_CUDA_ARCH)
      message(STATUS "Using user defined CUDA Arch: " "${SAIGA_CUDA_ARCH}")
        CUDA_SELECT_NVCC_ARCH_FLAGS(SAIGA_CUDA_ARCH_FLAGS ${SAIGA_CUDA_ARCH})
    else ()
        message(STATUS "Using automatic CUDA Arch detection...")
        CUDA_SELECT_NVCC_ARCH_FLAGS(SAIGA_CUDA_ARCH_FLAGS Auto )
        set(SAIGA_CUDA_ARCH ${SAIGA_CUDA_ARCH_FLAGS_arches} PARENT_SCOPE)
    endif ()
    message(STATUS "SAIGA_CUDA_ARCH: ${SAIGA_CUDA_ARCH}")
    message(STATUS "SAIGA_CUDA_ARCH_FLAGS: ${SAIGA_CUDA_ARCH_FLAGS}")


    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif ()


set(CUDA_INCLUDES ${PACKAGE_INCLUDES})
set(CUDA_LIBS ${LIBS})
set(CUDA_TARGETS ${LIB_TARGETS})

