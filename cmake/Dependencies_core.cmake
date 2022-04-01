# Defines the following output variables:
#
# CORE_INCLUDES:    The list of required include directories
# CORE_LIBS:        The list of required libraries for link_target
# CORE_TARGETS:     The list of required targets
# MODULE_CORE:      True if all required dependencies are found.
#

unset(PACKAGE_INCLUDES)
unset(LIB_TARGETS)
unset(LIBS)
unset(MODULE_CORE)


# The submodule setup is defined in submodules/CMakeLists.txt
# After that we can just use the generated targets in our build
add_subdirectory(submodules/)

PackageHelperTarget(Eigen3::Eigen EIGEN3_FOUND)
PackageHelperTarget(glfw GLFW_FOUND)
PackageHelperTarget(zlibstatic ZLIB_FOUND)
PackageHelperTarget(png_static PNG_FOUND)
PackageHelperTarget(assimp ASSIMP_FOUND)
PackageHelperTarget(glog GLOG_FOUND)

#openmp
if (SAIGA_CXX_WCLANG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xclang -fopenmp")
    find_library(OMP_LIB libomp PATH_SUFFIXES lib)
    message(STATUS ${OMP_LIB})
    SET(LIBS ${LIBS} ${OMP_LIB})
else ()
    find_package(OpenMP REQUIRED)

    #PackageHelperTarget(OpenMP::OpenMP_CXX OPENMP_FOUND)
    PackageHelper(OpenMP ${OPENMP_FOUND} "${OPENMP_INCLUDE_DIRS}" "${OpenMP_CXX_LIB_NAMES}")
     if(OPENMP_FOUND)
       list(APPEND SAIGA_CXX_FLAGS ${OpenMP_CXX_FLAGS})
       list(APPEND SAIGA_LD_FLAGS ${OpenMP_CXX_FLAGS})
     endif()

    #        # This line doesn't work with nvcc + gcc8.3. Just uncomment it.
    #        if(SAIGA_CXX_GNU)
    #            set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    #        else()
    #            PackageHelperTarget(OpenMP::OpenMP_CXX OPENMP_FOUND)
    #        endif()
    #    endif()
endif ()


#dbghelp for crash.cpp
if (WIN32)
    SET(LIBS ${LIBS} DbgHelp)
endif (WIN32)

############# Optional Libraries ###############


if (SAIGA_WITH_OPENAL)
    #openal
    find_package(OpenAL QUIET)
    if (OPENAL_FOUND)
        SET(SAIGA_USE_OPENAL 1)
    endif ()
    PackageHelper(OpenAL ${OPENAL_FOUND} "${OPENAL_INCLUDE_DIR}" "${OPENAL_LIBRARY}")


    #alut
    find_package(ALUT QUIET)
    if (ALUT_FOUND)
        SET(SAIGA_USE_ALUT 1)
    endif ()
    PackageHelper(ALUT ${ALUT_FOUND} "${ALUT_INCLUDE_DIRS}" "${ALUT_LIBRARIES}")


    #opus
    find_package(Opus QUIET)
    if (OPUS_FOUND)
        SET(SAIGA_USE_OPUS 1)
    endif ()
    PackageHelper(Opus ${OPUS_FOUND} "${OPUS_INCLUDE_DIRS}" "${OPUS_LIBRARIES}")
endif ()

find_package(FreeImage QUIET)
PackageHelper(FreeImage ${FREEIMAGE_FOUND} "${FREEIMAGE_INCLUDE_DIRS}" "${FREEIMAGE_LIBRARIES}")
if (FREEIMAGE_FOUND)
    SET(SAIGA_USE_FREEIMAGE 1)
endif ()


if (SAIGA_WITH_OPENMESH)
#openmesh
find_package(OpenMesh QUIET)
if (OPENMESH_FOUND)
    SET(SAIGA_USE_OPENMESH 1)
endif ()
PackageHelper(OpenMesh ${OPENMESH_FOUND} "${OPENMESH_INCLUDE_DIRS}" "${OPENMESH_LIBRARIES}")
endif()

set(CORE_INCLUDES ${PACKAGE_INCLUDES})
set(CORE_LIBS ${LIBS})
set(CORE_TARGETS ${LIB_TARGETS})
set(MODULE_CORE 1)

#message(STATUS "${LIBS}")
