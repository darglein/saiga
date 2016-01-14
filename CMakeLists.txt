cmake_minimum_required(VERSION 2.8)



set(DEPENDENCIES_INCLUDE_DIR "" CACHE FILEPATH "Path to yourlib includes")

set(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH} ${DEPENDENCIES_INCLUDE_DIR}/include)
set(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH} ${DEPENDENCIES_INCLUDE_DIR}/usr/include)
include_directories(${DEPENDENCIES_INCLUDE_DIR}/include)
include_directories(${DEPENDENCIES_INCLUDE_DIR}/usr/include)
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} ${DEPENDENCIES_INCLUDE_DIR}/lib)
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} ${DEPENDENCIES_INCLUDE_DIR}/usr/lib)
LINK_DIRECTORIES(${DEPENDENCIES_INCLUDE_DIR}/lib)
LINK_DIRECTORIES(${DEPENDENCIES_INCLUDE_DIR}/usr/lib)
LINK_DIRECTORIES(${DEPENDENCIES_INCLUDE_DIR}/bin)
  
set(LIB_NAME saiga)

project (${LIB_NAME})
set(CMAKE_BUILD_TYPE Release)
#INCLUDE(FindPkgConfig)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

######### Show all warnings

if(MSVC)
  # Force to always compile with W4
  if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
  endif()
elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  # Update if necessary
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic")
endif()



set (LIBRARY_VERSION_MAJOR 1)
set (LIBRARY_VERSION_MINOR 0)


option (USE_SDL "Use SDL2" OFF) 
option (USE_GLFW "Use GLFW" ON) 
option (USE_NOISE "Use NOISE" OFF) 
option (USE_GLEW "Use GLEW" ON) 
option (USE_GLBINDING "Use GLBINDING" OFF) 
option (USE_OPENAL "Use OPENAL" OFF) 
option (USE_ALUT "Use ALUT" OFF) 
option (USE_ASSIMP "Use Assimp" OFF) 
option (USE_PNG "Use PNG" OFF) 
option (USE_FREEIMAGE "Use FREEIMAGE" ON) 

option (BUILD_SHARED "Build this library as a shared/dynamic library" ON) 

configure_file (
  "${CMAKE_CURRENT_SOURCE_DIR}/${LIB_NAME}/buildconfig.h.in"
  "${CMAKE_CURRENT_SOURCE_DIR}/${LIB_NAME}/buildconfig.h"
)

############# Optional Libraries ###############


#SDL2
if (USE_SDL)
 	find_package(SDL2 REQUIRED)
	include_directories(${SDL2_INCLUDE_DIR})
	SET(LIBS ${LIBS} ${SDL2_LIBRARY})
endif (USE_SDL)


#GLFW
if (USE_GLFW)
	set(GLFW_LOCATION /home/um96udad/test/data)
  find_package(GLFW REQUIRED)
	include_directories(${GLFW_INCLUDE_DIR})
	SET(LIBS ${LIBS} ${GLFW_LIBRARIES})
	message(STATUS "glfw include: ${GLFW_INCLUDE_DIR} glfw libraries: ${GLFW_LIBRARIES}")
endif (USE_GLFW)

#libnoise
if (USE_NOISE)
	find_library(NOISE_LIBRARIES noise)
	SET(LIBS ${LIBS} ${NOISE_LIBRARIES})
endif (USE_NOISE)

#openal
if(USE_OPENAL)
	find_package(OpenAL REQUIRED)
	include_directories(${OPENAL_INCLUDE_DIRS})
	SET(LIBS ${LIBS} ${OPENAL_LIBRARY})
endif(USE_OPENAL)

#alut
if(USE_ALUT)
	find_package(ALUT REQUIRED)
	include_directories(${ALUT_INCLUDE_DIRS})
	SET(LIBS ${LIBS} ${ALUT_LIBRARIES})
endif(USE_ALUT)

#assimp
if(USE_ASSIMP)
	find_package(ASSIMP REQUIRED)
	include_directories(${ASSIMP_INCLUDE_DIRS})
	SET(LIBS ${LIBS} ${ASSIMP_LIBRARIES})
	#SET(LIBS ${LIBS} assimp)
endif(USE_ASSIMP)

#png
if(USE_PNG)
	find_package(PNG REQUIRED)
	include_directories(${PNG_INCLUDE_DIRS})
	SET(LIBS ${LIBS} ${PNG_LIBRARIES})
endif(USE_PNG)

#libfreeimage
if(USE_FREEIMAGE)
	find_package(FreeImage REQUIRED)
	include_directories(${FREEIMAGE_INCLUDE_PATH})
	SET(LIBS ${LIBS} ${FREEIMAGE_LIBRARY})

	find_package(FreeImagePlus REQUIRED)
	include_directories(${FREEIMAGEPLUS_INCLUDE_PATH})
	SET(LIBS ${LIBS} ${FREEIMAGEPLUS_LIBRARIES})
endif(USE_FREEIMAGE)

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

#GLEW
if(USE_GLEW)
	find_package(GLEW REQUIRED)
	include_directories(${GLEW_INCLUDE_DIRS} )
	SET(LIBS ${LIBS} ${GLEW_LIBRARIES})
endif(USE_GLEW)

#glbinding
if(USE_GLBINDING)
	find_package(glbinding REQUIRED)
	include_directories(${GLBINDING_INCLUDE_DIRS} )
	SET(LIBS ${LIBS} ${GLBINDING_LIBRARIES})
endif(USE_GLBINDING)








############# COMPILER FLAGS ###############


if(UNIX)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fvisibility=hidden -fvisibility-inlines-hidden")
endif(UNIX)


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

############# SOURCE ###############

add_subdirectory(src)
add_subdirectory(saiga)
add_subdirectory(shader)



