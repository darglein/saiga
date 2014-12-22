cmake_minimum_required(VERSION 2.8)

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

############# COMPILER FLAGS ###############

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

############# SOURCE ###############

add_subdirectory(src)
add_subdirectory(libhello)



