cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

set(LIB_NAME saiga)
project (${LIB_NAME} CXX)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH} )
include(helper_macros)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

set (LIBRARY_VERSION_MAJOR 1)
set (LIBRARY_VERSION_MINOR 1)

# Turn on the ability to create folders to organize projects (.vcproj)
# It creates "CMakePredefinedTargets" folder by default and adds CMake
# defined projects like INSTALL.vcproj and ZERO_CHECK.vcproj
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

############# Build Options ###############

message("Build Options")


#option (SAIGA_BUILD_SHARED "Build this library as a shared library" ON) 
OptionsHelper (SAIGA_BUILD_SHARED "Build this library as a shared library" ON) 

OptionsHelper (SAIGA_DEBUG "alot of error checks and more console output" ON) 
OptionsHelper (SAIGA_ASSERTS "enable the SAIGA_ASSERT makro" ON) 
OptionsHelper (SAIGA_BUILD_SAMPLES "build samples" ON) 
OptionsHelper (SAIGA_WITH_CUDA "use cuda if possible" ON) 
OptionsHelper (SAIGA_STRICT_FP "strict ieee floating point" OFF) 
OptionsHelper (SAIGA_FULL_OPTIMIZE "finds and enables all possible optimizations" OFF) 
OptionsHelper (SAIGA_CUDA_DEBUG "enables CUDA_SYNC_CHECK_ERROR and CUDA_ASSERT" ON) 
OptionsHelper(SAIGA_BUILD_DOC "Build documentation" ON)

DefaultBuildType()

message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

############# COMPILER FLAGS ###############


include(Saiga_compiler_flags)


############# Optional Libraries ###############


message("\nSearching for installed libaries...")

### SDL2 ###
find_package(SDL2 QUIET)
if (SDL2_FOUND)
	SET(SAIGA_USE_SDL 1)
endif()
PackageHelper(SDL2 ${SDL2_FOUND} "${SDL2_INCLUDE_DIRS}" "${SDL2_LIBRARIES}")


#GLFW
find_package(GLFW QUIET)
if (GLFW_FOUND)
	SET(SAIGA_USE_GLFW 1)
endif ()
PackageHelper(GLFW ${GLFW_FOUND} "${GLFW_INCLUDE_DIR}" "${GLFW_LIBRARIES}")


#EGL
find_package(EGL QUIET)
if(EGL_FOUND)
	SET(SAIGA_USE_EGL 1)
endif ()
PackageHelper(EGL ${EGL_FOUND} "${EGL_INCLUDE_DIRS}" "${EGL_LIBRARIES}")

#FFMPEG
find_package(FFMPEG REQUIRED)
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
find_package(ASSIMP QUIET)
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
find_package(Eigen3 REQUIRED)
if(EIGEN3_FOUND)
	SET(SAIGA_USE_EIGEN 1)
endif()
PackageHelper(Eigen3 "${EIGEN3_FOUND}" "${EIGEN3_INCLUDE_DIRS}" "")


#bullet
find_package(Bullet QUIET)
if(BULLET_FOUND)
	SET(SAIGA_USE_BULLET 1)
endif()
PackageHelper(Bullet "${BULLET_FOUND}" "${BULLET_INCLUDE_DIR}" "${BULLET_LIBRARIES}")



############# CUDA ###############

#Check Language is an extra module so we need to include it.
include(CheckLanguage)


if(SAIGA_WITH_CUDA)
	#find_package(CUDA QUIET)
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
	find_package(CUDA QUIET)
	SET(ALL_CUDA_LIBS ${CUDA_LIBRARIES} ${CUDA_cusparse_LIBRARY} ${CUDA_cublas_LIBRARY})
	#SET(LIBS ${LIBS} ${ALL_CUDA_LIBS})
#		SET(LIBS ${LIBS} ${CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES})
	message(STATUS "CUDA_LIBRARIES: ${ALL_CUDA_LIBS}")
	
	#message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
	message(STATUS "CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
	
	#message(STATUS "CUDA_LIBRARIES: ${ALL_CUDA_LIBS}")
	
	

	#Use cuda includes for host, too.
		include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
		#include_directories(${CUDA_INCLUDE_DIRS})

	#set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
	#set(CUDA_PROPAGATE_HOST_FLAGS ON)
	#set(CUDA_SEPARABLE_COMPILATION OFF)

	#list( APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_30,code=compute_30 )
	#list( APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_52,code=sm_52 )

	#list( APPEND CUDA_NVCC_FLAGS -use_fast_math)


	#disables an eigen warning
	#list( APPEND CMAKE_CUDA_FLAGS --expt-relaxed-constexpr)
	#list( APPEND CUDA_NVCC_FLAGS --expt-relaxed-constexpr)

	
	SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_30,code=compute_30")
	SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_52,code=sm_52")
	SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_61,code=sm_61")
	SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math --expt-relaxed-constexpr")
	
	#message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
	
	if(BUILD_SHARED)
	      #set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcompiler -DSAIGA_DLL_EXPORTS)
		  SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -DSAIGA_DLL_EXPORTS")
	endif()


	
	message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
message(STATUS "CMAKE_CUDA_FLAGS_DEBUG: ${CMAKE_CUDA_FLAGS_DEBUG}")
message(STATUS "CMAKE_CUDA_FLAGS_RELWITHDEBINFO: ${CMAKE_CUDA_FLAGS_RELWITHDEBINFO}")
message(STATUS "CMAKE_CUDA_FLAGS_RELEASE: ${CMAKE_CUDA_FLAGS_RELEASE}")

	SET(SAIGA_USE_CUDA 1)
else()
	SET(SAIGA_USE_CUDA 0)
	#message(STATUS "No CUDA.")
endif()

PackageHelper(CUDA "${CUDA_FOUND}" "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}" "${ALL_CUDA_LIBS}")




############# Required LIBRARIES ###############

#freetype2
find_package(Freetype REQUIRED QUIET)
PackageHelper(Freetype "${FREETYPE_FOUND}" "${FREETYPE_INCLUDE_DIRS}" "${FREETYPE_LIBRARIES}")

#OPENGL
find_package(OpenGL REQUIRED)
PackageHelper(OpenGL "${OPENGL_FOUND}" "${OPENGL_INCLUDE_DIR}" "${OPENGL_LIBRARIES}")

#GLM
find_package(GLM REQUIRED QUIET)
PackageHelper(GLM "${GLM_FOUND}" "${GLM_INCLUDE_DIRS}" "")

#glbinding
find_package(glbinding QUIET)
if(GLBINDING_FOUND)
	SET(SAIGA_USE_GLBINDING 1)
else()
	#use GLEW as a fallback
	find_package(GLEW REQUIRED) 
	if(GLEW_FOUND)
		SET(SAIGA_USE_GLEW 1)
	else()
		message(FATAL_ERROR "No OpenGL loading library found.")
	endif()
	PackageHelper(GLEW "${GLEW_FOUND}" "${GLEW_INCLUDE_DIRS}" "${GLEW_LIBRARIES}")
endif()
PackageHelper(glbinding "${GLBINDING_FOUND}" "${GLBINDING_INCLUDE_DIRS}" "${GLBINDING_LIBRARIES}")


#dbghelp for crash.cpp
if(WIN32)
	SET(LIBS ${LIBS} DbgHelp)
endif(WIN32)



############# INSTALL PATHS ###############

#This part is partly copied from the CMakeLists.txt of glbinding.
#If a relative path is given it is interpreted relative to the value of the CMAKE_INSTALL_PREFIX variable.
if(UNIX)

    # default CMAKE_INSTALL_PREFIX = /usr/local/
    set(INSTALL_ROOT      "share/${LIB_NAME}")
    set(INSTALL_SHADER    "share/${LIB_NAME}/shader")
    set(INSTALL_BIN       "bin")
    set(INSTALL_SHARED    "lib")
    set(INSTALL_LIB       "lib")
    set(INSTALL_INCLUDE   "include/${LIB_NAME}")
    set(INSTALL_DOC       "share/doc/${LIB_NAME}")

elseif(WIN32) 

    # default CMAKE_INSTALL_PREFIX = "%PROGRAMFILES%/<project>/"
    set(INSTALL_ROOT      ".")
    set(INSTALL_SHADER    "share/${LIB_NAME}/shader")
    set(INSTALL_BIN       "bin")
    set(INSTALL_SHARED    ".")
    set(INSTALL_LIB       "lib")
    set(INSTALL_INCLUDE   "include/${LIB_NAME}")
    set(INSTALL_DOC       "share/doc/${LIB_NAME}")

endif()

message("\nCompiler Flags:")
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "CMAKE_CXX_FLAGS_RELWITHDEBINFO: ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")
#message(STATUS "CMAKE_MODULE_LINKER_FLAGS: ${CMAKE_MODULE_LINKER_FLAGS}")
#message(STATUS "CMAKE_SHARED_LINKER_FLAGS: ${CMAKE_SHARED_LINKER_FLAGS}")
#message(STATUS "CMAKE_STATIC_LINKER_FLAGS: ${CMAKE_STATIC_LINKER_FLAGS}")
#message(STATUS "CMAKE_EXE_LINKER_FLAGS: ${CMAKE_EXE_LINKER_FLAGS}")

############# SOURCE ###############


message("")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")

add_subdirectory(src/saiga)
#add_subdirectory(include/saiga)
add_subdirectory(shader)

if(SAIGA_BUILD_SAMPLES)
	message(STATUS "Building samples.")
	add_subdirectory(samples)
else()
	message(STATUS "No samples.")
endif()

set_target_properties (saiga PROPERTIES FOLDER lib)

############# Generate config file ###############

configure_file (
  "${CMAKE_CURRENT_SOURCE_DIR}/src/${LIB_NAME}/buildconfig.h.in"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/${LIB_NAME}/buildconfig.h"
)



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




# Request that saiga be built with --std=c++11
# As this is a public compile feature anything that links to saiga
# will also build with -std=c++11
target_compile_features(${LIB_NAME} PUBLIC cxx_std_11)



set_target_properties(${LIB_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

