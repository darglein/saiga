cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

set(LIB_NAME saiga)
set(LIB_EXPORT saiga-export)

project (${LIB_NAME} CXX)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake" ${CMAKE_MODULE_PATH} )
include(helper_macros)



set (SAIGA_VERSION_MAJOR 1)
set (SAIGA_VERSION_MINOR 3)
set (SAIGA_VERSION_PATCH 2)
set(SAIGA_VERSION ${SAIGA_VERSION_MAJOR}.${SAIGA_VERSION_MINOR}.${SAIGA_VERSION_PATCH})

# Turn on the ability to create folders to organize projects (.vcproj)
# It creates "CMakePredefinedTargets" folder by default and adds CMake
# defined projects like INSTALL.vcproj and ZERO_CHECK.vcproj
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

enable_testing()

############# Build Options ###############

message(STATUS "\nBuild Options")


#option (SAIGA_BUILD_SHARED "Build this library as a shared library" ON)
OptionsHelper (SAIGA_BUILD_SHARED "Build this library as a shared library" ON)

OptionsHelper (SAIGA_USE_SUBMODULES "alot of error checks and more console output" OFF)
OptionsHelper (SAIGA_DEBUG "alot of error checks and more console output" OFF)
OptionsHelper (SAIGA_ASSERTS "enable the SAIGA_ASSERT makro" ON)
OptionsHelper (SAIGA_BUILD_SAMPLES "build samples" ON)
OptionsHelper (SAIGA_BUILD_TESTS "build tests" ON)
OptionsHelper (SAIGA_STRICT_FP "strict ieee floating point" OFF)
OptionsHelper (SAIGA_FAST_MATH "enable fast-math compiler flag" OFF)
OptionsHelper (SAIGA_FULL_OPTIMIZE "finds and enables all possible optimizations" OFF)
OptionsHelper (SAIGA_ARCHNATIVE "adds the -march=native compile flag" ON)
OptionsHelper (SAIGA_CUDA_DEBUG "enables CUDA_SYNC_CHECK_ERROR and CUDA_ASSERT" ON)
OptionsHelper (SAIGA_CUDA_BLSP "enables cublas and cusparse" OFF)
OptionsHelper (SAIGA_CUDA_RDC "cuda relocateable device code" OFF)
OptionsHelper (SAIGA_CUDA_COMP "Add CUDA compile flags that make the code compatible with older GPUs but takes longer to compile" OFF)
OptionsHelper(SAIGA_BUILD_DOC "Build documentation" OFF)
OptionsHelper(SAIGA_NO_INSTALL "The build directory can be used for external projects." OFF)
OptionsHelper(SAIGA_LEGACY_GLM "Use GLM instead of eigen. This feature will be removed in the near future" OFF)
OptionsHelper(SAIGA_PCH "Generate a precompiled header" OFF)
OptionsHelper(SAIGA_OPENMP "Enable OPENMP" ON)
OptionsHelper(SAIGA_LIBSTDCPP "Use the GCC std lib for the clang compiler" OFF)
OptionsHelper(SAIGA_DEBUG_ASAN "Enable the address sanitizer. Does not work in combination with TSAN." OFF)
OptionsHelper(SAIGA_DEBUG_MSAN "Enable the memory sanitizer. Does not work in combination with TSAN." OFF)
OptionsHelper(SAIGA_DEBUG_TSAN "Enable the thread sanitizer. Does not work in combination with ASAN." OFF)
OptionsHelper(SAIGA_DEBIAN_BUILD "Saiga is currently build into a Debian Package. This will overwrite a lot of compile flags." OFF)


OptionsHelper(SAIGA_MODULE_CORE   "Build the core module, if possible." ON)
OptionsHelper(SAIGA_MODULE_OPENGL "Build the opengl module, if possible." ON)
OptionsHelper(SAIGA_MODULE_VULKAN "Build the vulkan module, if possible." ON)
OptionsHelper(SAIGA_MODULE_VISION "Build the vision module, if possible." ON)
OptionsHelper(SAIGA_MODULE_CUDA   "Build the cuda module, if possible." ON)

OptionsHelper(SAIGA_WITH_MKL "Build the MKL library, if possible." OFF)


if(SAIGA_DEBIAN_BUILD)
    # Force release mode for package builds
    # Debug info will make the files too big.
    set(CMAKE_BUILD_TYPE Release)
    set(SAIGA_ARCHNATIVE OFF)
    set(SAIGA_FULL_OPTIMIZE OFF)
endif()

DefaultBuildType(RelWithDebInfo)

add_definitions()


set(EXTERNAL_DIR ${PROJECT_SOURCE_DIR}/External)


set(SAIGA_FULL_EIGEN 1)
if(SAIGA_LEGACY_GLM)
    set(SAIGA_FULL_EIGEN 0)
endif()

############# OUTPUT ###############

set(SAIGA_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}")

# Make to cache variable so it can be overwritten by the user of saiga
set(SAIGA_RUNTIME_OUTPUT_DIRECTORY "${SAIGA_OUTPUT_DIR}/bin" CACHE PATH "Output dir of executables and dynamic libraries")
set(SAIGA_LIBRARY_OUTPUT_DIRECTORY "${SAIGA_OUTPUT_DIR}/lib" CACHE PATH "Output dir of static libraries")
set(SAIGA_ARCHIVE_OUTPUT_DIRECTORY "${SAIGA_OUTPUT_DIR}/lib" CACHE PATH "Output dir of static libraries")


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "${SAIGA_RUNTIME_OUTPUT_DIRECTORY}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  "${SAIGA_LIBRARY_OUTPUT_DIRECTORY}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY  "${SAIGA_ARCHIVE_OUTPUT_DIRECTORY}")



############# COMPILER FLAGS ###############

include(Saiga_compiler_flags)

####### Dependencies #######
# Build Submodules first



unset(MODULE_CORE)
unset(MODULE_CUDA)
unset(MODULE_EXTRA)
unset(MODULE_OPENGL)
unset(MODULE_VISION)
unset(MODULE_VULKAN)

if(SAIGA_MODULE_CORE)
    include (Dependencies_core)
endif()

if(MODULE_CORE)
    if(SAIGA_MODULE_OPENGL)
        include (Dependencies_opengl)
    endif()
    if(SAIGA_MODULE_VISION)
        include (Dependencies_vision)
    endif()
    if(SAIGA_MODULE_VULKAN)
        include (Dependencies_vulkan)
    endif()
    if(SAIGA_MODULE_CUDA)
        include (Dependencies_cuda)
    endif()
endif()

unset(PACKAGE_INCLUDES)
unset(LIB_TARGETS)
unset(LIBS)

############# INSTALL PATHS ###############

#This part is partly copied from the CMakeLists.txt of glbinding.
#If a relative path is given it is interpreted relative to the value of the CMAKE_INSTALL_PREFIX variable.
#if(UNIX)

set(INSTALL_INCLUDE   "include/${LIB_NAME}")
set(INSTALL_BIN       "bin")
set(INSTALL_SHARED    "lib")
set(INSTALL_LIB       "lib")
set(INSTALL_SHARE     "share/${LIB_NAME}")
set(INSTALL_SHADER    "share/${LIB_NAME}/shader")
set(INSTALL_DOC       "share/${LIB_NAME}/doc")
set(INSTALL_SAMPLES   "share/${LIB_NAME}/bin")



message(STATUS "\nCompiler Flags:")

message(STATUS "SAIGA_CXX_FLAGS: ${SAIGA_CXX_FLAGS}")
message(STATUS "SAIGA_PRIVATE_CXX_FLAGS: ${SAIGA_PRIVATE_CXX_FLAGS}")
message(STATUS "SAIGA_LD_FLAGS: ${SAIGA_LD_FLAGS}")

message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "CMAKE_CXX_FLAGS_RELWITHDEBINFO: ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")

if(CUDA_FOUND)
    message(STATUS "\nCUDA Compiler Flags:")
    message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
    message(STATUS "CMAKE_CUDA_FLAGS_DEBUG: ${CMAKE_CUDA_FLAGS_DEBUG}")
    message(STATUS "CMAKE_CUDA_FLAGS_RELWITHDEBINFO: ${CMAKE_CUDA_FLAGS_RELWITHDEBINFO}")
    message(STATUS "CMAKE_CUDA_FLAGS_RELEASE: ${CMAKE_CUDA_FLAGS_RELEASE}")
endif()



############# Generate config file ###############

set(SAIGA_PROJECT_SOURCE_DIR ${PROJECT_SOURCE_DIR})

set(SAIGA_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX} )

if(SAIGA_DEBIAN_BUILD)
    # Note during a debian build the actual install dir is something like /libsaiga0.0.8/debian/tmp/usr
    # We set it here to /usr so the programs can find the shader later.
    set(SAIGA_INSTALL_PREFIX /usr)
    # The source is not included in the final package
    set(SAIGA_PROJECT_SOURCE_DIR "-")
endif()

if(SAIGA_NO_INSTALL)
    set(SAIGA_SHADER_PREFIX "${PROJECT_SOURCE_DIR}/shader")
    set(SAIGA_INCLUDE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}/src")
    set(SAIGA_LIB_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/bin")
    set(SAIGA_BUILD_INCLUDE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/include")
    set(SAIGA_INSTALL_PREFIX "-")
    set(SAIGA_CONFIG_CMAKE_FILE_NO_INSTALL "${CMAKE_CURRENT_BINARY_DIR}/SAIGAConfig.cmake")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/SAIGAConfigNoInstall.cmake.in"
        "${SAIGA_CONFIG_CMAKE_FILE_NO_INSTALL}"
        @ONLY
        )
else()
    set(SAIGA_SHADER_PREFIX "${SAIGA_INSTALL_PREFIX}/share/saiga/shader")
    set(SAIGA_INCLUDE_INSTALL_PREFIX "${SAIGA_INSTALL_PREFIX}/include")
    set(SAIGA_BUILD_INCLUDE_INSTALL_PREFIX "${SAIGA_INSTALL_PREFIX}/include")
    set(SAIGA_LIB_INSTALL_PREFIX "${SAIGA_INSTALL_PREFIX}/lib")
endif()

STRING(REGEX REPLACE "\\\\" "/" SAIGA_SHADER_PREFIX ${SAIGA_SHADER_PREFIX} )
STRING(REGEX REPLACE "\\\\" "/" SAIGA_INSTALL_PREFIX ${SAIGA_INSTALL_PREFIX} )

## Generate source file with git commit
include(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC SAIGA_GIT_SHA1)

set(SAIGA_BUILDCONFIG_DIR "${CMAKE_CURRENT_BINARY_DIR}/include")
set(SAIGA_BUILDCONFIG_FILE "${SAIGA_BUILDCONFIG_DIR}/saiga/saiga_buildconfig.h")
set(SAIGA_MODULES_FILE "${SAIGA_BUILDCONFIG_DIR}/saiga/saiga_modules.h")
set(SAIGA_GIT_FILE "${SAIGA_BUILDCONFIG_DIR}/saiga/saiga_git_sha1.h")
set(SAIGA_CONFIGURED_FILES ${SAIGA_BUILDCONFIG_FILE} ${SAIGA_MODULES_FILE} ${SAIGA_GIT_FILE})
#INCLUDE_DIRECTORIES(${SAIGA_BUILDCONFIG_DIR})
configure_file (
    "${CMAKE_CURRENT_SOURCE_DIR}/src/${LIB_NAME}/saiga_buildconfig.h.in"
    "${SAIGA_BUILDCONFIG_FILE}"
    )
configure_file (
    "${CMAKE_CURRENT_SOURCE_DIR}/src/${LIB_NAME}/saiga_modules.h.in"
    "${SAIGA_MODULES_FILE}"
    )
configure_file (
        "${CMAKE_CURRENT_SOURCE_DIR}/src/${LIB_NAME}/saiga_git_sha1.h.in"
        "${SAIGA_GIT_FILE}"
        @ONLY
)



set(SAIGA_CONFIG_CMAKE_FILE "${CMAKE_CURRENT_BINARY_DIR}/cmake/SAIGAConfig.cmake")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/SAIGAConfig.cmake.in"
    "${SAIGA_CONFIG_CMAKE_FILE}"
    @ONLY
    )




# Generate Version File
set(SAIGA_VERSION_CMAKE_FILE "${CMAKE_CURRENT_BINARY_DIR}/cmake/SAIGAConfigVersion.cmake")
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${SAIGA_VERSION_CMAKE_FILE}
    VERSION ${SAIGA_VERSION}
    COMPATIBILITY AnyNewerVersion
    )




############# SOURCE ###############


add_subdirectory(src)
add_subdirectory(shader)

if(SAIGA_BUILD_SAMPLES)
    message(STATUS " ")
    add_subdirectory(samples)
    message(STATUS " ")
else()
    message(STATUS "\nNo samples.")
endif()

if(SAIGA_BUILD_TESTS)
  message(STATUS " ")

  if(SAIGA_USE_SUBMODULES)
      set(GTEST_TARGETS gtest gtest_main)
  else(NOT SAIGA_USE_SUBMODULES)
    find_package(GTest CONFIG QUIET)
    if (TARGET GTest::gtest)
      PackageHelperTarget(GTest::gtest GTEST_FOUND)
      set(GTEST_TARGETS GTest::gtest GTest::gtest_main)
    else()
      find_package(GTest QUIET)
      PackageHelperTarget(GTest::gtest GTEST_FOUND)
      set(GTEST_TARGETS GTest::gtest GTest::gtest_main)
    endif()
  endif()

  add_subdirectory(tests)

  message(STATUS " ")
else()
  message(STATUS "\nNo tests.")
endif()

#set_target_properties (saiga PROPERTIES FOLDER lib)

############# INSTALL ###############

#install config file to share/
install(
    FILES "${SAIGA_CONFIG_CMAKE_FILE}" "${SAIGA_VERSION_CMAKE_FILE}"
    DESTINATION ${INSTALL_SHARE}
    )

#install buildconfig include/
install(
    FILES "${SAIGA_BUILDCONFIG_FILE}"
    DESTINATION ${INSTALL_INCLUDE}
    )

install(
    FILES "${SAIGA_MODULES_FILE}"
    DESTINATION ${INSTALL_INCLUDE}
    )

install(
    DIRECTORY ./src/saiga/
    DESTINATION ${INSTALL_INCLUDE}
    FILES_MATCHING PATTERN "*.h"
    PATTERN "*.inl"
    PATTERN "*.hpp"
    )


############# UNINSTALL ###############

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/uninstall.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/uninstall.cmake
    IMMEDIATE @ONLY)

add_custom_target(uninstall_saiga
    COMMAND ${CMAKE_COMMAND} -P
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/uninstall.cmake)
	
set_target_properties(uninstall_saiga PROPERTIES FOLDER CMakePredefinedTargets)

############# Doxygen ###############

# first we can indicate the documentation build as an option and set it to ON by default


# check if Doxygen is installed
find_package(Doxygen)
if (DOXYGEN_FOUND AND SAIGA_BUILD_DOC)
    # set input and output files
    set(DOXYGEN_IN ${CMAKE_CURRENT_LIST_DIR}/docs/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
    set(DOXYGEN_MAIN ${CMAKE_CURRENT_LIST_DIR}/docs/main.md)
    # request to configure the file
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    message(STATUS "Doxygen found. Generate with 'make saiga_doc'.")

    # note the option ALL which allows to build the docs together with the application
    add_custom_target( saiga_doc
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM )

    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html)

    install(
        DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html
        DESTINATION ${INSTALL_DOC}
        FILES_MATCHING PATTERN "*"

        )
endif()








