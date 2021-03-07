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


# ========= Libraries that are included as submodules =========
find_package(Eigen3 3.3.90 QUIET)
if(NOT TARGET Eigen3::Eigen)
  message("=================================")
  message("Adding Submodule eigen")
   set(BUILD_TESTING OFF CACHE INTERNAL "")
  add_subdirectory(submodules/eigen)
  message("=================================")
endif()
SET(SAIGA_USE_EIGEN 1)
PackageHelperTarget(Eigen3::Eigen EIGEN3_FOUND)



find_package(glfw3 CONFIG )
if(NOT TARGET glfw)
  message("=================================")
  message("Adding Submodule glfw")
  set(BUILD_SHARED_LIBS ON CACHE INTERNAL "")
  set(GLFW_BUILD_EXAMPLES OFF CACHE INTERNAL "")
  set(GLFW_BUILD_TESTS OFF CACHE INTERNAL "")
  add_subdirectory(submodules/glfw)
  message("=================================")
endif ()
PackageHelperTarget(glfw GLFW_FOUND)
SET(SAIGA_USE_GLFW 1)


#zlib
find_package(ZLIB QUIET)
if(${ZLIB_FOUND})
  add_library(zlib INTERFACE)
  target_link_libraries(zlib INTERFACE "${ZLIB_LIBRARIES}")
  target_include_directories(zlib INTERFACE "${ZLIB_INCLUDE_DIRS}")
  PackageHelperTarget(zlib ZLIB_FOUND)
  SET(SAIGA_USE_ZLIB 1)
elseif(SAIGA_USE_SUBMODULES)
  message("=================================")
  message("Building Submodule ZLIB")
  add_subdirectory(submodules/zlib)
  PackageHelperTarget(zlib ZLIB_FOUND)
  SET(SAIGA_USE_ZLIB 1)
  message("=================================")
endif()

# png
find_package(PNG QUIET)
if(${PNG_FOUND})
  PackageHelper(PNG ${PNG_FOUND} "${PNG_INCLUDE_DIRS}" "${PNG_LIBRARIES}")
  SET(SAIGA_USE_PNG 1)
elseif(SAIGA_USE_SUBMODULES)
  message("=================================")
  message("Building Submodule libPNG")
  set(PNG_BUILD_ZLIB ON CACHE INTERNAL "")
  set(PNG_STATIC OFF CACHE INTERNAL "")
  set(PNG_EXECUTABLES OFF CACHE INTERNAL "")
  set(PNG_TESTS OFF CACHE INTERNAL "")
  set(ZLIB_LIBRARIES zlib CACHE INTERNAL "")
  add_subdirectory(submodules/libpng)
  PackageHelperTarget(png PNG_FOUND)
  SET(SAIGA_USE_PNG 1)
  message("=================================")
endif()


#assimp
find_package(ASSIMP QUIET)
if(ASSIMP_FOUND)
  PackageHelper(ASSIMP ${ASSIMP_FOUND} "${ASSIMP_INCLUDE_DIRS}" "${ASSIMP_LIBRARIES}")
  SET(SAIGA_USE_ASSIMP 1)
elseif(SAIGA_USE_SUBMODULES)
  message("=================================")
  message("Building Submodule assimp")
  add_subdirectory(submodules/assimp)
  PackageHelperTarget(assimp ASSIMP_FOUND)
  SET(SAIGA_USE_ASSIMP 1)
endif()


#openmp
if(SAIGA_CXX_WCLANG)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xclang -fopenmp")
  find_library(OMP_LIB libomp PATH_SUFFIXES lib)
  message(STATUS ${OMP_LIB})
  SET(LIBS ${LIBS} ${OMP_LIB})
else()
  find_package(OpenMP REQUIRED)

  PackageHelperTarget(OpenMP::OpenMP_CXX OPENMP_FOUND)
  # PackageHelper(OpenMP ${OPENMP_FOUND} "${OPENMP_INCLUDE_DIRS}" "${OPENMP_LIBRARIES}")
  # if(OPENMP_FOUND)
  #   list(APPEND SAIGA_CXX_FLAGS ${OpenMP_CXX_FLAGS})
  #   list(APPEND SAIGA_LD_FLAGS ${OpenMP_CXX_FLAGS})
  # endif()

  #        # This line doesn't work with nvcc + gcc8.3. Just uncomment it.
  #        if(SAIGA_CXX_GNU)
  #            set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  #        else()
  #            PackageHelperTarget(OpenMP::OpenMP_CXX OPENMP_FOUND)
  #        endif()
  #    endif()
endif()


#dbghelp for crash.cpp
if(WIN32)
  SET(LIBS ${LIBS} DbgHelp)
endif(WIN32)

############# Optional Libraries ###############




#GLFW
#find_package(GLFW 3.2 QUIET)
#PackageHelper(GLFW ${GLFW_FOUND} "${GLFW_INCLUDE_DIR}" "${GLFW_LIBRARIES}")


#target_link_libraries(main PRIVATE glfw)


#openal
find_package(OpenAL QUIET)
if(OPENAL_FOUND)
  SET(SAIGA_USE_OPENAL 1)
endif()
PackageHelper(OpenAL ${OPENAL_FOUND} "${OPENAL_INCLUDE_DIR}" "${OPENAL_LIBRARY}")


#alut
find_package(ALUT QUIET)
if(ALUT_FOUND)
  SET(SAIGA_USE_ALUT 1)
endif()
PackageHelper(ALUT ${ALUT_FOUND} "${ALUT_INCLUDE_DIRS}" "${ALUT_LIBRARIES}")


#opus
find_package(Opus QUIET)
if(OPUS_FOUND)
  SET(SAIGA_USE_OPUS 1)
endif()
PackageHelper(Opus ${OPUS_FOUND} "${OPUS_INCLUDE_DIRS}" "${OPUS_LIBRARIES}")



#libfreeimage
find_package(FreeImagePlus QUIET)
PackageHelper(FreeImagePlus ${FREEIMAGEPLUS_FOUND} "${FREEIMAGEPLUS_INCLUDE_PATH}" "${FREEIMAGEPLUS_LIBRARIES}")
find_package(FreeImage QUIET)
PackageHelper(FreeImage ${FREEIMAGE_FOUND} "${FREEIMAGE_INCLUDE_PATH}" "${FREEIMAGE_LIBRARIES}")
if(FREEIMAGE_FOUND AND FREEIMAGEPLUS_FOUND)
  SET(SAIGA_USE_FREEIMAGE 1)
endif()



#c++17 filesystem
#find_package(Filesystem REQUIRED QUIET)
#SET(SAIGA_USE_FILESYSTEM 1)
#PackageHelperTarget(std::filesystem FILESYSTEM_FOUND)
#if(FILESYSTEM_FOUND)
#endif()


#openmesh
find_package(OpenMesh QUIET)
if(OPENMESH_FOUND)
  SET(SAIGA_USE_OPENMESH 1)
endif()
PackageHelper(OpenMesh ${OPENMESH_FOUND} "${OPENMESH_INCLUDE_DIRS}" "${OPENMESH_LIBRARIES}")


set(CORE_INCLUDES ${PACKAGE_INCLUDES})
set(CORE_LIBS ${LIBS})
set(CORE_TARGETS ${LIB_TARGETS})
set(MODULE_CORE 1)

#message(STATUS "${LIBS}")
