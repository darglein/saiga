cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

set(LIB_NAME saiga)
set(LIB_EXPORT saiga-export)

project (${LIB_NAME} CXX)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH} )
include(helper_macros)



set (SAIGA_VERSION_MAJOR 1)
set (SAIGA_VERSION_MINOR 2)
set (SAIGA_VERSION_PATCH 0)
set(SAIGA_VERSION ${SAIGA_VERSION_MAJOR}.${SAIGA_VERSION_MINOR}.${SAIGA_VERSION_PATCH})

# Turn on the ability to create folders to organize projects (.vcproj)
# It creates "CMakePredefinedTargets" folder by default and adds CMake
# defined projects like INSTALL.vcproj and ZERO_CHECK.vcproj
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

############# Build Options ###############

message(STATUS "\nBuild Options")


#option (SAIGA_BUILD_SHARED "Build this library as a shared library" ON)
OptionsHelper (SAIGA_BUILD_SHARED "Build this library as a shared library" ON)

OptionsHelper (SAIGA_DEBUG "alot of error checks and more console output" ON)
OptionsHelper (SAIGA_ASSERTS "enable the SAIGA_ASSERT makro" ON)
OptionsHelper (SAIGA_BUILD_SAMPLES "build samples" ON)
OptionsHelper (SAIGA_WITH_CUDA "use cuda if possible" ON)
OptionsHelper (SAIGA_STRICT_FP "strict ieee floating point" OFF)
OptionsHelper (SAIGA_FULL_OPTIMIZE "finds and enables all possible optimizations" OFF)
OptionsHelper (SAIGA_CUDA_DEBUG "enables CUDA_SYNC_CHECK_ERROR and CUDA_ASSERT" ON)
OptionsHelper (SAIGA_CUDA_BLSP "enables cublas and cusparse" OFF)
OptionsHelper (SAIGA_CUDA_RDC "cuda relocateable device code" OFF)
OptionsHelper (SAIGA_CUDA_COMP "Add CUDA compile flags that make the code compatible with older GPUs but takes longer to compile" OFF)
OptionsHelper(SAIGA_BUILD_DOC "Build documentation" ON)
OptionsHelper(SAIGA_WITH_GL "With opengl (if it is found)" ON)
OptionsHelper(SAIGA_WITH_VULKAN "With vulkan (if it is found)" ON)
OptionsHelper(SAIGA_U_EIGEN_CUDA "Eigen CUDA Support" OFF)
OptionsHelper(SAIGA_NO_INSTALL "The build directory can be used for external projects." OFF)
OptionsHelper(SAIGA_FULL_EIGEN "Use the eigen library for CG math instead of GLM" OFF)
OptionsHelper(SAIGA_PCH "Generate a precompiled header" OFF)
OptionsHelper(SAIGA_LIBSTDCPP "Use the GCC std lib for the clang compiler" OFF)


OptionsHelper(SAIGA_MODULE_CORE   "Build the core module, if possible." ON)
OptionsHelper(SAIGA_MODULE_OPENGL "Build the opengl module, if possible." ON)
OptionsHelper(SAIGA_MODULE_VULKAN "Build the vulkan module, if possible." ON)
OptionsHelper(SAIGA_MODULE_EXTRA  "Build the extra module, if possible." ON)
OptionsHelper(SAIGA_MODULE_VISION "Build the vision module, if possible." ON)
OptionsHelper(SAIGA_MODULE_CUDA   "Build the cuda module, if possible." ON)


DefaultBuildType(RelWithDebInfo)


############# COMPILER FLAGS ###############


include(Saiga_compiler_flags)


if(SAIGA_PCH)
    include(cotire)
endif()

############# Required LIBRARIES ###############

message(STATUS "\nSearching for installed libaries...")

#GLM
find_package(GLM REQUIRED QUIET)
PackageHelper(GLM "${GLM_FOUND}" "${GLM_INCLUDE_DIRS}" "")


#dbghelp for crash.cpp
if(WIN32)
    SET(LIBS ${LIBS} DbgHelp)
endif(WIN32)

############# Optional Libraries ###############



# SDL2
find_package(SDL2 QUIET)
if (SDL2_FOUND)
    SET(SAIGA_USE_SDL 1)
endif()
PackageHelper(SDL2 ${SDL2_FOUND} "${SDL2_INCLUDE_DIR}" "${SDL2_LIBRARY}")


#GLFW
find_package(GLFW 3.2 QUIET)
if (GLFW_FOUND)
    SET(SAIGA_USE_GLFW 1)
endif ()
PackageHelper(GLFW ${GLFW_FOUND} "${GLFW_INCLUDE_DIR}" "${GLFW_LIBRARIES}")

#message(STATUS ${GLFW_LIBRARIES})


#EGL
find_package(EGL QUIET)
if(EGL_FOUND)
    SET(SAIGA_USE_EGL 1)
endif ()
PackageHelper(EGL ${EGL_FOUND} "${EGL_INCLUDE_DIRS}" "${EGL_LIBRARIES}")

#FFMPEG
find_package(FFMPEG QUIET)
if (FFMPEG_FOUND)
    SET(SAIGA_USE_FFMPEG 1)
endif ()
PackageHelper(FFMPEG ${FFMPEG_FOUND} "${FFMPEG_INCLUDE_DIR}" "${FFMPEG_LIBRARIES}")

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

#assimp
find_package(ASSIMP 4.0 QUIET)
if(ASSIMP_FOUND)
    SET(SAIGA_USE_ASSIMP 1)
endif()
PackageHelper(ASSIMP ${ASSIMP_FOUND} "${ASSIMP_INCLUDE_DIRS}" "${ASSIMP_LIBRARIES}")

#png
find_package(PNG QUIET)
if(PNG_FOUND)
    SET(SAIGA_USE_PNG 1)
endif()
PackageHelper(PNG ${PNG_FOUND} "${PNG_INCLUDE_DIRS}" "${PNG_LIBRARIES}")

#openmesh
find_package(OpenMesh QUIET)
if(OPENMESH_FOUND)
    SET(SAIGA_USE_OPENMESH 1)
endif()
PackageHelper(OpenMesh ${OPENMESH_FOUND} "${OPENMESH_INCLUDE_DIRS}" "${OPENMESH_LIBRARIES}")

#openmp
find_package(OpenMP QUIET)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
PackageHelper(OpenMP ${OPENMP_FOUND} "${OPENMP_INCLUDE_DIRS}" "${OPENMP_LIBRARIES}")

#libfreeimage
find_package(FreeImage QUIET)
PackageHelper(FreeImage ${FREEIMAGE_FOUND} "${FREEIMAGE_INCLUDE_PATH}" "${FREEIMAGE_LIBRARIES}")
find_package(FreeImagePlus QUIET)
PackageHelper(FreeImagePlus ${FREEIMAGEPLUS_FOUND} "${FREEIMAGEPLUS_INCLUDE_PATH}" "${FREEIMAGEPLUS_LIBRARIES}")

if(FREEIMAGE_FOUND AND FREEIMAGEPLUS_FOUND)
    SET(SAIGA_USE_FREEIMAGE 1)
endif()




#eigen
find_package(Eigen3 QUIET)
PackageHelperTarget(Eigen3::Eigen EIGEN3_FOUND)
if(EIGEN3_FOUND)
    SET(SAIGA_USE_EIGEN 1)
endif()

#Sophus
find_package(Sophus QUIET)
PackageHelperTarget(Sophus::Sophus SOPHUS_FOUND)

#g2o
find_package(g2o QUIET)
PackageHelperTarget(g2o::core G2O_FOUND)


#ceres
find_package(Ceres QUIET)
if(CERES_FOUND)
    SET(SAIGA_USE_CERES 1)
endif()
PackageHelperTarget(ceres CERES_FOUND)


#bullet
find_package(Bullet QUIET)
if(BULLET_FOUND)
    SET(SAIGA_USE_BULLET 1)
endif()
PackageHelper(Bullet "${BULLET_FOUND}" "${BULLET_INCLUDE_DIR}" "${BULLET_LIBRARIES}")


# Currently only mkl is supported
#set(BLA_VENDOR Intel10_64lp_seq)
#find_package(BLAS QUIET)
#PackageHelper(BLAS "${BLAS_FOUND}" "" "${BLAS_LIBRARIES}")
#message(STATUS "BLAS Library: ${BLAS_LIBRARIES}")

#find_package(LAPACK QUIET)
#PackageHelper(LAPACK "${LAPACK_FOUND}" "" "${LAPACK_LIBRARIES}")
#message(STATUS "LAPACK Library: ${LAPACK_LIBRARIES}")

#mkl
find_package(MKL QUIET)
if(MKL_FOUND )
    SET(SAIGA_USE_MKL 1)
endif()
PackageHelper(MKL "${MKL_FOUND}" "${MKL_INCLUDE_DIR}" "${MKL_LIBRARIES}")



#openni2
find_package(OpenNI2 QUIET)
if(OPENNI2_FOUND)
    SET(SAIGA_USE_OPENNI2 1)
endif()
PackageHelper(OpenNI2 "${OPENNI2_FOUND}" "${OPENNI2_INCLUDE_DIRS}" "${OPENNI2_LIBRARIES}")


find_package(Boost QUIET COMPONENTS thread system)
if(Boost_FOUND)
    SET(SAIGA_USE_BOOST 1)
endif()
PackageHelper(Boost "${Boost_FOUND}" "${Boost_INCLUDE_DIRS}" "${Boost_LIBRARIES}")


#opencv
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    SET(SAIGA_USE_OPENCV 1)
endif()
PackageHelper(OpenCV "${OpenCV_FOUND}" "${OpenCV_INCLUDE_DIRS}" "${OpenCV_LIBRARIES}")


##### Vulkan #####
if(SAIGA_MODULE_OPENGL)
    find_package(Vulkan QUIET)
    PackageHelperTarget(Vulkan::Vulkan VULKAN_FOUND)

    find_package(GLslang QUIET)
    PackageHelper(GLslang "${GLSLANG_FOUND}" "${GLSLANG_SPIRV_INCLUDE_DIR}" "${GLSLANG_LIBRARIES}")
    if(${VULKAN_FOUND} AND ${GLSLANG_FOUND})
        message(STATUS "Saiga vulkan enabled.")
        SET(SAIGA_USE_VULKAN 1)
    endif()
else()
    UNSET(SAIGA_USE_VULKAN)
endif()



if(SAIGA_MODULE_OPENGL)
    set(OpenGL_GL_PREFERENCE LEGACY)
    find_package(OpenGL QUIET)
    PackageHelper(OpenGL "${OPENGL_FOUND}" "${OPENGL_INCLUDE_DIR}" "${OPENGL_LIBRARIES}")
    if(OPENGL_FOUND)
        SET(SAIGA_USE_OPENGL 1)
    endif()
else()
    UNSET(SAIGA_USE_OPENGL)
endif()



#freetype2
find_package(Freetype QUIET)
PackageHelper(Freetype "${FREETYPE_FOUND}" "${FREETYPE_INCLUDE_DIRS}" "${FREETYPE_LIBRARIES}")
if(FREETYPE_FOUND)
    SET(SAIGA_USE_FREETYPE 1)
endif()

#freetype2
find_package(Gphoto2 QUIET)
PackageHelper(Gphoto2 "${GPHOTO2_FOUND}" "${Gphoto2_INCLUDE_DIRS}" "${Gphoto2_LIBRARIES}")
if(GPHOTO2_FOUND)
    SET(SAIGA_USE_GPHOTO2 1)
endif()

#c++17 filesystem
find_package(Filesystem QUIET)
PackageHelper(Filesystem "${FILESYSTEM_FOUND}" "" "${FILESYSTEM_LIBRARY}")
if(FILESYSTEM_FOUND)
    SET(SAIGA_USE_FILESYSTEM 1)
endif()


############# CUDA ###############

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
    find_package(CUDA REQUIRED QUIET)

    SET(ALL_CUDA_LIBS ${CUDA_LIBRARIES})

    if(SAIGA_CUDA_BLSP)
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
    SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_61,code=sm_61")

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

    if(BUILD_SHARED)
        SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -DSAIGA_DLL_EXPORTS")
    endif()
    SET(SAIGA_USE_CUDA 1)

    if(SAIGA_U_EIGEN_CUDA)
        #if(${Eigen3_VERSION} VERSION_GREATER_EQUAL "3.3.90" AND ${CUDA_VERSION} VERSION_GREATER_EQUAL "9.2")
        SET(SAIGA_EIGEN_AND_CUDA 1)
        #endif()
    endif()
else()
    SET(SAIGA_USE_CUDA 0)
endif()

PackageHelper(CUDA "${CUDA_FOUND}" "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}" "${ALL_CUDA_LIBS}")



if(SAIGA_EIGEN_AND_CUDA)
    message(STATUS "Enabled Eigen with CUDA -> Eigen ${Eigen3_VERSION} Cuda ${CUDA_VERSION}")
endif()


if(EIGEN3_FOUND AND SOPHUS_FOUND)
    message(STATUS "Saiga vision enabled.")
    SET(SAIGA_VISION 1)
endif()


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

if(ajshgsdgasdgdsg)
    set(INSTALL_SHARE     "share/${LIB_NAME}")
    set(INSTALL_SHADER    "share/${LIB_NAME}/shader")
    set(INSTALL_BIN       "bin")
    set(INSTALL_SHARED    ".")
    set(INSTALL_LIB       "lib")
    set(INSTALL_INCLUDE   "include/${LIB_NAME}")
    set(INSTALL_DOC       "share/doc/${LIB_NAME}")

endif()


message(STATUS "\nCompiler Flags:")
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

if(SAIGA_NO_INSTALL)
    set(SAIGA_SHADER_PREFIX "${PROJECT_SOURCE_DIR}/shader")

    set(SAIGA_CONFIG_CMAKE_FILE_NO_INSTALL "${CMAKE_CURRENT_BINARY_DIR}/SAIGAConfig.cmake")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/SAIGAConfigNoInstall.cmake.in"
        "${SAIGA_CONFIG_CMAKE_FILE_NO_INSTALL}"
        @ONLY
        )
else()
    set(SAIGA_SHADER_PREFIX "${CMAKE_INSTALL_PREFIX}/share/saiga/shader")
endif()

set(SAIGA_BUILDCONFIG_DIR "${CMAKE_CURRENT_BINARY_DIR}/include")
set(SAIGA_BUILDCONFIG_FILE "${SAIGA_BUILDCONFIG_DIR}/saiga_buildconfig.h")
set(SAIGA_MODULES_FILE "${SAIGA_BUILDCONFIG_DIR}/saiga_modules.h")
INCLUDE_DIRECTORIES(${SAIGA_BUILDCONFIG_DIR})
configure_file (
    "${CMAKE_CURRENT_SOURCE_DIR}/src/${LIB_NAME}/saiga_buildconfig.h.in"
    "${SAIGA_BUILDCONFIG_FILE}"
    )
configure_file (
    "${CMAKE_CURRENT_SOURCE_DIR}/src/${LIB_NAME}/saiga_modules.h.in"
    "${SAIGA_MODULES_FILE}"
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

############# C++ Standard and Filesystem stuff ###############

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


############# SOURCE ###############

set(OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/bin")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "${OUTPUT_DIR}")



add_subdirectory(src)
add_subdirectory(shader)

if(SAIGA_BUILD_SAMPLES)
    message(STATUS " ")
    add_subdirectory(samples)
    message(STATUS " ")
else()
    message(STATUS "\nNo samples.")
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



############# UNINSTALL ###############

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/uninstall.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/uninstall.cmake
    IMMEDIATE @ONLY)

add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/uninstall.cmake)

############# Doxygen ###############

# first we can indicate the documentation build as an option and set it to ON by default


# check if Doxygen is installed
find_package(Doxygen)
if (DOXYGEN_FOUND AND SAIGA_BUILD_DOC)
    # set input and output files
    set(DOXYGEN_IN ${CMAKE_SOURCE_DIR}/docs/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
    set(DOXYGEN_MAIN ${CMAKE_SOURCE_DIR}/docs/main.md)
    # request to configure the file
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    message(STATUS "Doxygen found. Generate with 'make doc'.")

    # note the option ALL which allows to build the docs together with the application
    add_custom_target( doc
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




# Request that saiga be built with --std=c++14
# As this is a public compile feature anything that links to saiga
# will also build with -std=c++14

# This currently does not work with cuda
#target_compile_features(${LIB_NAME} PUBLIC cxx_std_17)

#set_property(TARGET ${LIB_NAME} PROPERTY CXX_STANDARD 17)
#set_property(TARGET ${LIB_NAME} PROPERTY CUDA_STANDARD 14)






