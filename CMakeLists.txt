cmake_minimum_required(VERSION 2.8)

set(GRALAB_INCLUDE_DIR "" CACHE FILEPATH "Path to yourlib includes")
include_directories(${GRALAB_INCLUDE_DIR}/include)
include_directories(${GRALAB_INCLUDE_DIR}/usr/include)
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} ${GRALAB_INCLUDE_DIR}/lib ${GRALAB_INCLUDE_DIR}/usr/lib)
LINK_DIRECTORIES(${GRALAB_INCLUDE_DIR}/lib)
LINK_DIRECTORIES(${GRALAB_INCLUDE_DIR}/usr/lib)

set(LIB_NAME hello)


project (${LIB_NAME})
set(CMAKE_BUILD_TYPE Release)
INCLUDE(FindPkgConfig)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/libhello)

############# LIBRARIES ###############

#freetype2
find_package(Freetype REQUIRED)
include_directories(${FREETYPE_INCLUDE_DIRS})

#SDL2
PKG_SEARCH_MODULE(SDL2 REQUIRED sdl2)
include_directories(${SDL2_INCLUDE_DIRS})

#GLFW
pkg_search_module(GLFW REQUIRED glfw3)
include_directories(${GLFW_INCLUDE_DIRS})

#libnoise
find_library(NOISE_LIBRARIES noise)
SET(LIBS ${LIBS} ${NOISE_LIBRARIES})

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



