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

if(SAIGA_USE_SUBMODULES)
  message("=================================")
  message("Adding Submodule eigen")
   set(BUILD_TESTING OFF CACHE INTERNAL "")
  add_subdirectory(submodules/eigen)
  message("=================================")
else()
find_package(Eigen3 3.3.90 QUIET REQUIRED)
endif()
SET(SAIGA_USE_EIGEN 1)
PackageHelperTarget(Eigen3::Eigen EIGEN3_FOUND)


if(SAIGA_USE_SUBMODULES)
  message("=================================")
  message("Adding Submodule glfw")
  set(BUILD_SHARED_LIBS ON CACHE INTERNAL "")
  set(GLFW_BUILD_EXAMPLES OFF CACHE INTERNAL "")
  set(GLFW_BUILD_TESTS OFF CACHE INTERNAL "")
  add_subdirectory(submodules/glfw)
  set_target_properties(glfw PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${SAIGA_RUNTIME_OUTPUT_DIRECTORY}")
  message("=================================")
else()
  find_package(glfw3 CONFIG QUIET )
endif ()
PackageHelperTarget(glfw GLFW_FOUND)
SET(SAIGA_USE_GLFW 1)




if(SAIGA_USE_SUBMODULES)
  message("=================================")
  message("Adding Submodule ZLIB")
  add_subdirectory(submodules/zlib)
  set(ZLIB_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../submodules/zlib" CACHE PATH "zlib dir" FORCE)
  target_include_directories(zlib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../submodules/zlib>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/submodules/zlib>  )

   set_target_properties(zlibstatic PROPERTIES EXCLUDE_FROM_ALL 1)
  PackageHelperTarget(zlib ZLIB_FOUND)
  set_target_properties(zlib PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${SAIGA_RUNTIME_OUTPUT_DIRECTORY}")
  #message(FATAL_ERROR ${ZLIB_INCLUDE_DIR})
  SET(SAIGA_USE_ZLIB 1)
  message("=================================")
else()
  find_package(ZLIB QUIET)
    add_library(zlib INTERFACE)
    target_link_libraries(zlib INTERFACE "${ZLIB_LIBRARIES}")
    target_include_directories(zlib INTERFACE "${ZLIB_INCLUDE_DIRS}")
    PackageHelperTarget(zlib ZLIB_FOUND)
    SET(SAIGA_USE_ZLIB 1)
endif()

# png
  if(SAIGA_USE_SUBMODULES)
  message("=================================")
  message("Adding Submodule libPNG")

  set(PNG_BUILD_ZLIB ON CACHE INTERNAL "")
  set(PNG_STATIC OFF CACHE INTERNAL "")
  set(PNG_EXECUTABLES OFF CACHE INTERNAL "")
  set(PNG_TESTS OFF CACHE INTERNAL "")
  set(ZLIB_LIBRARIES zlib CACHE INTERNAL "")
  set(SKIP_INSTALL_ALL ON CACHE INTERNAL "")

  include_directories(${ZLIB_INCLUDE_DIRS})
  add_subdirectory(submodules/libpng)

    target_include_directories(png PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../submodules/libpng>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/submodules/libpng>  )


  PackageHelperTarget(png PNG_FOUND)
  set_target_properties(png PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${SAIGA_RUNTIME_OUTPUT_DIRECTORY}")
  SET(SAIGA_USE_PNG 1)

  set(CMAKE_INSTALL_LIBDIR lib)
    install(TARGETS png zlib
          EXPORT libpng
          RUNTIME DESTINATION bin
          LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
          ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
          FRAMEWORK DESTINATION ${CMAKE_INSTALL_LIBDIR})

  install(EXPORT libpng
          DESTINATION lib/libpng)

  message("=================================")
    else()

    find_package(PNG QUIET)
    PackageHelper(PNG ${PNG_FOUND} "${PNG_INCLUDE_DIRS}" "${PNG_LIBRARIES}")
    SET(SAIGA_USE_PNG 1)
endif()


#assimp
find_package(ASSIMP QUIET)
if(ASSIMP_FOUND)
  PackageHelper(ASSIMP ${ASSIMP_FOUND} "${ASSIMP_INCLUDE_DIRS}" "${ASSIMP_LIBRARIES}")
  SET(SAIGA_USE_ASSIMP 1)
elseif(SAIGA_USE_SUBMODULES)
  message("=================================")
  message("Adding Submodule assimp")

  set(ASSIMP_BUILD_TESTS OFF CACHE INTERNAL "")
  set(ASSIMP_BUILD_ASSIMP_TOOLS OFF CACHE INTERNAL "")

  add_subdirectory(submodules/assimp)
  PackageHelperTarget(assimp ASSIMP_FOUND)
  set_target_properties(assimp PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${SAIGA_RUNTIME_OUTPUT_DIRECTORY}")
  SET(SAIGA_USE_ASSIMP 1)
  message("=================================")
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
