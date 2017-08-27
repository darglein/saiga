cmake_minimum_required(VERSION 2.8)

set(LIB_NAME saiga)
project (${LIB_NAME})

set(DEPENDENCIES_DIR "" CACHE FILEPATH "Path to your lib includes")

set(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH} ${DEPENDENCIES_DIR}/include)
set(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH} ${DEPENDENCIES_DIR}/usr/include)
include_directories(${DEPENDENCIES_DIR}/include)
include_directories(${DEPENDENCIES_DIR}/usr/include)
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} ${DEPENDENCIES_DIR}/lib)
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} ${DEPENDENCIES_DIR}/usr/lib)
LINK_DIRECTORIES(${DEPENDENCIES_DIR}/lib)
LINK_DIRECTORIES(${DEPENDENCIES_DIR}/usr/lib)
LINK_DIRECTORIES(${DEPENDENCIES_DIR}/bin)
  

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH} )
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

######### Show all warnings

if(MSVC)
  # Force to always compile with W4
  if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
  endif()
elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long")
endif()



set (LIBRARY_VERSION_MAJOR 1)
set (LIBRARY_VERSION_MINOR 0)



option (BUILD_SHARED "Build this library as a shared/dynamic library" ON) 



############# Build Options ###############

option (BUILD_MODE_DEBUG "Defines SAIGA_DEBUG, asserts=on, debug symbols in binary, opengl debug context, alot of glgeterror calls" OFF) 
option (BUILD_MODE_TESTING "Defines SAIGA_TESTING, asserts=on, debug symbols in binary, opengl debug context, one glgeterror per frame, (default)" ON) 
option (BUILD_MODE_RELEASE "Defines SAIGA_RELEASE, asserts=off, no glgeterror calls" OFF) 
option (BUILD_SAMPLES "build samples" ON) 
option (CUDA_DEBUG "cuda debug flag" ON) 


#in Multi-configuration IDEs (visual studio) setting CMAKE_BUILD_TYPE does not have a effect
if (BUILD_MODE_DEBUG)
	set(SAIGA_DEBUG on)
	set(CMAKE_BUILD_TYPE debug)
	#adds this flags on gcc: -g
endif ()

if (BUILD_MODE_TESTING)
	set(SAIGA_TESTING on)
	set(CMAKE_BUILD_TYPE relwithdebinfo)
	#adds this flags on gcc: -O2 -g -DNDEBUG
endif()

if (BUILD_MODE_RELEASE)
	set(SAIGA_RELEASE on)
	set(CMAKE_BUILD_TYPE release)
	#adds this flags on gcc: -O3 -DNDEBUG
endif()



#TODO build options for windows


############# Optional Libraries ###############


#SDL2
find_package(SDL2 QUIET)
if (SDL2_FOUND)
	include_directories(${SDL2_INCLUDE_DIR})
	SET(LIBS ${LIBS} ${SDL2_LIBRARY})
	SET(SAIGA_USE_SDL 1)
endif ()


#GLFW
find_package(GLFW QUIET)
if (GLFW_FOUND)
	include_directories(${GLFW_INCLUDE_DIR})
	SET(LIBS ${LIBS} ${GLFW_LIBRARIES})
	if(NOT( ${GLFW_VERSION_MAJOR} GREATER 2 AND ${GLFW_VERSION_MINOR} GREATER 1)  )
	  message( FATAL_ERROR "ERROR: Need at least GLFW version 3.2. -- found GLFW version ${GLFW_VERSION_MAJOR}.${GLFW_VERSION_MINOR}.${GLFW_VERSION_REVISION}" )
	endif()
	SET(SAIGA_USE_GLFW 1)
endif ()

#EGL
find_package(EGL QUIET)
if(EGL_FOUND)
	include_directories( ${EGL_INCLUDE_DIRS}) 
	SET(LIBS ${LIBS} ${EGL_LIBRARIES})
	SET(SAIGA_USE_EGL 1)
endif()


#FFMPEG
find_package(FFMPEG QUIET)
if (FFMPEG_FOUND)
	include_directories(${FFMPEG_INCLUDE_DIR})
	SET(LIBS ${LIBS} ${FFMPEG_LIBRARIES})
	SET(SAIGA_USE_FFMPEG 1)
endif ()

#libnoise
find_library(NOISE_LIBRARIES noise)
if (NOISE_FOUND)
	SET(LIBS ${LIBS} ${NOISE_LIBRARIES})
	SET(SAIGA_USE_NOISE 1)
endif ()

#openal
find_package(OpenAL QUIET)
if(OPENAL_FOUND)
	include_directories(${OPENAL_INCLUDE_DIRS})
	SET(LIBS ${LIBS} ${OPENAL_LIBRARY})
	SET(SAIGA_USE_OPENAL 1)
endif()


#alut
find_package(ALUT QUIET)
if(ALUT_FOUND)
	include_directories(${ALUT_INCLUDE_DIRS})
	SET(LIBS ${LIBS} ${ALUT_LIBRARIES})
	SET(SAIGA_USE_ALUT 1)
endif()


#opus
find_package(Opus QUIET)
if(OPUS_FOUND)
	include_directories(${OPUS_INCLUDE_DIRS})
	SET(LIBS ${LIBS} ${OPUS_LIBRARIES})
	SET(SAIGA_USE_OPUS 1)
	message(STATUS "Found Opus: ${OPUS_LIBRARIES}")
endif()

#assimp
find_package(ASSIMP QUIET)
if(ASSIMP_FOUND)
	include_directories(${ASSIMP_INCLUDE_DIRS})
	SET(LIBS ${LIBS} ${ASSIMP_LIBRARIES})
	SET(SAIGA_USE_ASSIMP 1)
	message(STATUS "Found ASSIMP: ${ASSIMP_LIBRARIES}")
endif()

#png
find_package(PNG QUIET)
if(PNG_FOUND)
	include_directories(${PNG_INCLUDE_DIRS})
	SET(LIBS ${LIBS} ${PNG_LIBRARIES})
	SET(SAIGA_USE_PNG 1)
	message(STATUS "Found libpng: ${PNG_LIBRARIES}")
endif()

#libfreeimage
find_package(FreeImage QUIET)
find_package(FreeImagePlus QUIET)

if(FREEIMAGE_FOUND AND FREEIMAGEPLUS_FOUND)
	include_directories(${FREEIMAGE_INCLUDE_PATH})
	SET(LIBS ${LIBS} ${FREEIMAGE_LIBRARIES})
	include_directories(${FREEIMAGEPLUS_INCLUDE_PATH})
	SET(LIBS ${LIBS} ${FREEIMAGEPLUS_LIBRARIES})
	SET(SAIGA_USE_FREEIMAGE 1)
	message(STATUS "Found FreeImage: ${FREEIMAGE_LIBRARIES} ${FREEIMAGEPLUS_LIBRARIES}")
endif()



#eigen
find_package(Eigen)
if(EIGEN_FOUND)
	include_directories(${EIGEN_INCLUDE_DIRS})
	SET(SAIGA_USE_EIGEN 1)
	message(STATUS "Found Eigen3: ${EIGEN_INCLUDE_DIRS}")
endif()


############# CUDA ###############

find_package(CUDA QUIET)
if(CUDA_FOUND)
	include_directories(${CUDA_INCLUDE_DIRS})
	SET(ALL_CUDA_LIBS ${CUDA_LIBRARIES} ${CUDA_cusparse_LIBRARY} ${CUDA_cublas_LIBRARY})
	SET(LIBS ${LIBS} ${ALL_CUDA_LIBS})
	message(STATUS "CUDA_LIBRARIES: ${CUDA_INCLUDE_DIRS} ${ALL_CUDA_LIBS}")

	#set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
	set(CUDA_PROPAGATE_HOST_FLAGS ON)
	set(CUDA_SEPARABLE_COMPILATION OFF)

	list( APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_30,code=compute_30 )
	list( APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_52,code=sm_52 )

	list( APPEND CUDA_NVCC_FLAGS -use_fast_math)

	#disables an eigen warning
	list( APPEND CUDA_NVCC_FLAGS --expt-relaxed-constexpr)

	if(BUILD_SHARED)
	      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcompiler -DSAIGA_DLL_EXPORTS)
	endif()

	if(UNIX)
	      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcompiler -fPIC)
	endif()

	SET(SAIGA_USE_CUDA 1)
endif()




message(STATUS "NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")
message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")


############# Required LIBRARIES ###############

#freetype2
find_package(Freetype REQUIRED)
include_directories(${FREETYPE_INCLUDE_DIRS})
SET(LIBS ${LIBS} ${FREETYPE_LIBRARIES})

if(WIN32)
SET(LIBS ${LIBS} freetype255)
endif(WIN32)

#OPENGL
find_package(OpenGL REQUIRED)
include_directories( ${OPENGL_INCLUDE_DIRS}) 
SET(LIBS ${LIBS} ${OPENGL_LIBRARIES})

#GLM
find_package(GLM REQUIRED)
include_directories( ${GLM_INCLUDE_DIRS}) 

#glbinding
find_package(glbinding QUIET)
if(GLBINDING_FOUND)
	include_directories(${GLBINDING_INCLUDE_DIRS} )
	SET(LIBS ${LIBS} ${GLBINDING_LIBRARIES})
	SET(SAIGA_USE_GLBINDING 1)
else()
#use GLEW as a fallback
#GLEW
find_package(GLEW QUIET)
if(GLEW_FOUND)
	include_directories(${GLEW_INCLUDE_DIRS} )
	SET(LIBS ${LIBS} ${GLEW_LIBRARIES})
	SET(SAIGA_USE_GLEW 1)
endif()
endif()



#dbghelp for crash.cpp
if(WIN32)
SET(LIBS ${LIBS} DbgHelp)
endif(WIN32)





############# COMPILER FLAGS ###############


if(UNIX)
#	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fvisibility=hidden -fvisibility-inlines-hidden")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fvisibility=hidden -msse2 -mfpmath=sse")
	if(CUDA_DEBUG)
	else()
		SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCUDA_NDEBUG")
	endif()
endif(UNIX)
if(MSVC)
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:SSE2 /fp:strict")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:strict")
endif()

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
    set(INSTALL_SHADER    "shader")
    set(INSTALL_BIN       "bin")
    set(INSTALL_SHARED    ".")
    set(INSTALL_LIB       "lib")
    set(INSTALL_INCLUDE   "include/${LIB_NAME}")
    set(INSTALL_DOC       "doc")

endif()

message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "CMAKE_CXX_FLAGS_RELWITHDEBINFO: ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "CMAKE_MODULE_LINKER_FLAGS: ${CMAKE_MODULE_LINKER_FLAGS}")
message(STATUS "CMAKE_SHARED_LINKER_FLAGS: ${CMAKE_SHARED_LINKER_FLAGS}")
message(STATUS "CMAKE_STATIC_LINKER_FLAGS: ${CMAKE_STATIC_LINKER_FLAGS}")
message(STATUS "CMAKE_EXE_LINKER_FLAGS: ${CMAKE_EXE_LINKER_FLAGS}")

############# SOURCE ###############

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")

add_subdirectory(src)
add_subdirectory(include/saiga)
add_subdirectory(shader)
if(BUILD_SAMPLES)
	message(STATUS "Building samples.")
	add_subdirectory(samples)
else()
	message(STATUS "No samples.")
endif()


############# Generate config file ###############

configure_file (
  "${CMAKE_CURRENT_SOURCE_DIR}/include/${LIB_NAME}/buildconfig.h.in"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/${LIB_NAME}/buildconfig.h"
)



