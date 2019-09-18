# Defines the following output variables:
#
# CUDA_INCLUDES:    The list of required include directories
# CUDA_LIBS:        The list of required libraries for link_target
# CUDA_TARGETS:     The list of required targets
# MODULE_CUDA:      True if all required dependencies are found.
#

unset(PACKAGE_INCLUDES)
unset(LIB_TARGETS)
unset(LIBS)
unset(MODULE_CUDA)

unset(CUDA_FOUND)
unset(CMAKE_CUDA_COMPILER)

#Check Language is an extra module so we need to include it.
include(CheckLanguage)

if(SAIGA_MODULE_CUDA)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CUDA_FOUND TRUE)
    else()
        set(CUDA_FOUND FALSE)
    endif()
else()
    set(CUDA_FOUND FALSE)
endif()



if(CUDA_FOUND)
    #message(STATUS "Found CUDA: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    # We need this to get the actual cuda libraries, because
    # one sample only is linked with the host compiler and therefore
    # does not automatically link to the cuda runtime.

    if(SAIGA_CUDA_BLSP)
        find_package(CUDA REQUIRED QUIET)

        SET(ALL_CUDA_LIBS ${CUDA_LIBRARIES})

        if(CUDA_cusparse_LIBRARY)
            SET(ALL_CUDA_LIBS ${ALL_CUDA_LIBS} ${CUDA_cusparse_LIBRARY})
            SET(SAIGA_USE_CUSPARSE 1)
        endif()

        if(CUDA_cublas_LIBRARY)
            SET(ALL_CUDA_LIBS ${ALL_CUDA_LIBS} ${CUDA_cublas_LIBRARY})
            SET(SAIGA_USE_CUBLAS 1)
        endif()
    endif()




    if(SAIGA_CUDA_COMP)
        SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_30,code=sm_30")
        SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_30,code=compute_30")
    endif()
    SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_52,code=sm_52")

    if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL "9")
        SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_61,code=sm_61") # Pascal
    endif()

    #SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_70,code=sm_70") # Volta

    if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL "10")
        SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_75,code=sm_75") # Turing
    endif()


    if(SAIGA_CUDA_RDC)
        SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --relocatable-device-code=true") # for dynamic parallelism
    else()

        if(SAIGA_CUDA_COMP AND SAIGA_CUDA_RDC)
            message(FATAL_ERROR "Invalid combination.")
        endif()

    endif()

    SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math --expt-relaxed-constexpr")
    #ignore warning "__device__ on defaulted function..."
    SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")

    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -stdlib=libstdc++")
    endif()

    if(BUILD_SHARED)
        SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -DSAIGA_DLL_EXPORTS")
    endif()
    SET(SAIGA_USE_CUDA 1)

    set(MODULE_CUDA 1)

else()
    SET(SAIGA_USE_CUDA 0)
    set(MODULE_CUDA 0)
endif()

#PackageHelper(CUDA "${CUDA_FOUND}" "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}" "${ALL_CUDA_LIBS}")



if(SAIGA_EIGEN_AND_CUDA)
    message(STATUS "Enabled Eigen with CUDA -> Eigen ${Eigen3_VERSION} Cuda ${CUDA_VERSION}")
endif()


set(CUDA_INCLUDES ${PACKAGE_INCLUDES})
set(CUDA_LIBS ${LIBS})
set(CUDA_TARGETS ${LIB_TARGETS})

