cmake_minimum_required(VERSION 2.8)



set(DEPENDENCIES_INCLUDE_DIR "" CACHE FILEPATH "Path to yourlib includes")
set(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH} ${DEPENDENCIES_INCLUDE_DIR}/include)
include_directories(${DEPENDENCIES_INCLUDE_DIR}/include)
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} ${DEPENDENCIES_INCLUDE_DIR}/lib)
LINK_DIRECTORIES(${DEPENDENCIES_INCLUDE_DIR}/lib)
LINK_DIRECTORIES(${DEPENDENCIES_INCLUDE_DIR}/bin)
  
set(LIB_NAME hello)

project (${LIB_NAME})
set(CMAKE_BUILD_TYPE Release)
#INCLUDE(FindPkgConfig)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/libhello)

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

############# Optional Libraries ###############

option (USE_SDL "Use SDL2" ON) 
option (USE_GLFW "Use GLFW" ON) 
option (USE_NOISE "Use NOISE" OFF) 

configure_file (
  "${CMAKE_CURRENT_SOURCE_DIR}/libhello/config.h.in"
  "${CMAKE_CURRENT_SOURCE_DIR}/libhello/config.h"
)


#SDL2
if (USE_SDL)
 	find_package(SDL2 REQUIRED)
	include_directories(${SDL2_INCLUDE_DIR})
	SET(LIBS ${LIBS} ${SDL2_LIBRARY})
endif (USE_SDL)

#GLFW
if (USE_GLFW)
  find_package(GLFW REQUIRED)
	include_directories(${GLFW_INCLUDE_DIR})
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

if(WIN32)
SET(LIBS ${LIBS} freetype255)
endif(WIN32)

#OPENGL
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLEW_INCLUDE_DIRS} )
SET(LIBS ${LIBS} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES})


#assimp
#pkg_search_module(ASSIMP REQUIRED assimp)
find_package(ASSIMP REQUIRED)
include_directories(${ASSIMP_INCLUDE_DIRS})
SET(LIBS ${LIBS} ${ASSIMP_LIBRARY})
SET(LIBS ${LIBS} assimp)



#libfreeimage
#pkg_search_module(FREEIMAGE REQUIRED freeimage)
#find_library(FREEIMAGE_LIBRARIES freeimage)
find_package(FreeImage REQUIRED)
SET(LIBS ${LIBS} ${FREEIMAGE_LIBRARY})

#find_library(FREEIMAGEPLUS_LIBRARIES freeimageplus)
find_package(FreeImagePlus REQUIRED)
SET(LIBS ${LIBS} ${FREEIMAGEPLUS_LIBRARIES})


if(UNIX)
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
endif(UNIX)


############# COMPILER FLAGS ###############
if(UNIX)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif(UNIX)

############# SOURCE ###############

add_subdirectory(src)
add_subdirectory(libhello)



