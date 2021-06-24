# Defines the following output variables:
#
# VISION_INCLUDES:    The list of required include directories
# VISION_LIBS:        The list of required libraries for link_target
# VISION_TARGETS:     The list of required targets
# MODULE_VISION:      True if all required dependencies are found.
#

include(ExternalProject)

unset(PACKAGE_INCLUDES)
unset(LIB_TARGETS)
unset(LIBS)
unset(MODULE_VISION)


if(NOT MODULE_CORE)
    return()
endif()


#opencv
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    SET(SAIGA_USE_OPENCV 1)
endif()
PackageHelper(OpenCV "${OpenCV_FOUND}" "${OpenCV_INCLUDE_DIRS}" "${OpenCV_LIBRARIES}")


set(CMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY ON)
set(CMAKE_EXPORT_NO_PACKAGE_REGISTRY ON)


#Sophus
find_package(Sophus QUIET)
PackageHelperTarget(Sophus::Sophus SOPHUS_FOUND)
if(SOPHUS_FOUND)
    SET(SAIGA_SYSTEM_SOPHUS 1)
endif()


# lib yaml for dataset loading
find_package(yaml-cpp QUIET)
PackageHelperTarget(yaml-cpp YAML_FOUND)
if(YAML_FOUND)
    SET(SAIGA_USE_YAML_CPP 1)
endif()

#Recursive
SET(SAIGA_USE_EIGENRECURSIVE 1)
find_package(EigenRecursive QUIET)
PackageHelperTarget(Eigen::EigenRecursive EIGENRECURSIVE_FOUND)
if(EIGENRECURSIVE_FOUND)
    SET(SAIGA_SYSTEM_EIGENRECURSIVE 1)
endif()

#g2o
find_package(g2o QUIET)
PackageHelperTarget(g2o::core G2O_FOUND)
if(G2O_FOUND)
    SET(SAIGA_USE_G2O 1)
endif()

#ceres
find_package(Ceres QUIET)
PackageHelperTarget(ceres CERES_FOUND)
if(CERES_FOUND)
    SET(SAIGA_USE_CERES 1)
endif()

#cholmod
find_package(CHOLMOD QUIET)
if(CHOLMOD_FOUND)
    SET(SAIGA_USE_CHOLMOD 1)
endif()
PackageHelper(CHOLMOD ${CHOLMOD_FOUND} "${CHOLMOD_INCLUDES}" "${CHOLMOD_LIBRARIES}")


# Currently only mkl is supported
#set(BLA_VENDOR Intel10_64lp_seq)
#find_package(BLAS QUIET)
#PackageHelper(BLAS "${BLAS_FOUND}" "" "${BLAS_LIBRARIES}")
#message(STATUS "BLAS Library: ${BLAS_LIBRARIES}")

#find_package(LAPACK QUIET)
#PackageHelper(LAPACK "${LAPACK_FOUND}" "" "${LAPACK_LIBRARIES}")
#message(STATUS "LAPACK Library: ${LAPACK_LIBRARIES}")

#mkl
if(SAIGA_WITH_MKL)
    find_package(MKL QUIET)
    if(MKL_FOUND )
        SET(SAIGA_USE_MKL 1)
    endif()
    PackageHelper(MKL "${MKL_FOUND}" "${MKL_INCLUDE_DIR}" "${MKL_LIBRARIES}")
endif()



#openni2
find_package(OpenNI2 QUIET)
if(OPENNI2_FOUND)
    SET(SAIGA_USE_OPENNI2 1)
endif()
PackageHelper(OpenNI2 "${OPENNI2_FOUND}" "${OPENNI2_INCLUDE_DIRS}" "${OPENNI2_LIBRARIES}")

#kinect azure sdk
find_package(k4a QUIET)
PackageHelperTarget(k4a::k4a K4A_FOUND)
if(K4A_FOUND)
    SET(SAIGA_USE_K4A 1)
endif()


set(VISION_INCLUDES ${PACKAGE_INCLUDES})
set(VISION_LIBS ${LIBS})
set(VISION_TARGETS saiga_core ${LIB_TARGETS})


message(STATUS "Saiga vision enabled.")
SET(MODULE_VISION 1)
SET(SAIGA_VISION 1)



