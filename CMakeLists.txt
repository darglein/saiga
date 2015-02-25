cmake_minimum_required(VERSION 2.8)

#set(GRALAB_INCLUDE_DIR "" CACHE FILEPATH "Path to yourlib includes")
#include_directories(${GRALAB_INCLUDE_DIR}/include)
#include_directories(${GRALAB_INCLUDE_DIR}/usr/include)
#set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} ${GRALAB_INCLUDE_DIR}/lib ${GRALAB_INCLUDE_DIR}/usr/lib)
#LINK_DIRECTORIES(${GRALAB_INCLUDE_DIR}/lib)
#LINK_DIRECTORIES(${GRALAB_INCLUDE_DIR}/usr/lib)

set(LIB_NAME hello)


project (${LIB_NAME})
set(CMAKE_BUILD_TYPE Release)
INCLUDE(FindPkgConfig)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/libhello)


############# Optional Libraries ###############

option (USE_SDL "Use SDL2" OFF) 
option (USE_GLFW "Use GLFW" ON) 
option (USE_NOISE "Use NOISE" OFF) 

configure_file (
  "${CMAKE_CURRENT_SOURCE_DIR}/libhello/config.h.in"
  "${CMAKE_CURRENT_SOURCE_DIR}/libhello/config.h"
)


#SDL2
if (USE_SDL)
	PKG_SEARCH_MODULE(SDL2 REQUIRED sdl2)
	include_directories(${SDL2_INCLUDE_DIRS})
	SET(LIBS ${LIBS} ${SDL2_LIBRARIES})
endif (USE_SDL)

#GLFW
if (USE_GLFW)
	pkg_search_module(GLFW REQUIRED glfw3)
	include_directories(${GLFW_INCLUDE_DIRS})
	SET(LIBS ${LIBS} ${GLFW_LIBRARIES})
endif (USE_GLFW)

#libnoise
if (USE_NOISE)
	find_library(NOISE_LIBRARIES noise)
	SET(LIBS ${LIBS} ${NOISE_LIBRARIES})
endif (USE_NOISE)

############# Required LIBRARIES ###############

#freetype2
find_package(Freetype REQUIRED)
include_directories(${FREETYPE_INCLUDE_DIRS})
SET(LIBS ${LIBS} ${FREETYPE_LIBRARIES})


#OPENGL
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLEW_INCLUDE_DIRS} )
SET(LIBS ${LIBS} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES})


#assimp
pkg_search_module(ASSIMP REQUIRED assimp)
include_directories(${ASSIMP_INCLUDE_DIRS})
SET(LIBS ${LIBS} ${ASSIMP_LIBRARIES})



#libfreeimage
#pkg_search_module(FREEIMAGE REQUIRED freeimage)
find_library(FREEIMAGE_LIBRARIES freeimage)
SET(LIBS ${LIBS} ${FREEIMAGE_LIBRARIES})

find_library(FREEIMAGEPLUS_LIBRARIES freeimageplus)
SET(LIBS ${LIBS} ${FREEIMAGEPLUS_LIBRARIES})


find_library(X11_LIBRARIES X11)
SET(LIBS ${LIBS} ${X11_LIBRARIES})

find_library(Xxf86vm_LIBRARIES Xxf86vm)
SET(LIBS ${LIBS} ${Xxf86vm_LIBRARIES})

find_library(Xrandr_LIBRARIES Xrandr)
SET(LIBS ${LIBS} ${Xrandr_LIBRARIES})

find_library(pthread_LIBRARIES pthread)
SET(LIBS ${LIBS} ${pthread_LIBRARIES})

find_library(Xi_LIBRARIES Xi)
SET(LIBS ${LIBS} ${Xi_LIBRARIES})

find_library(Xcursor_LIBRARIES Xcursor)
SET(LIBS ${LIBS} ${Xcursor_LIBRARIES})

find_library(Xinerama_LIBRARIES Xinerama)
SET(LIBS ${LIBS} ${Xinerama_LIBRARIES})


############# COMPILER FLAGS ###############

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

############# SOURCE ###############

add_subdirectory(src)
add_subdirectory(libhello)



