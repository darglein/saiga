message("=================================")
message("Adding Submodule eigen")
set(BUILD_TESTING OFF CACHE INTERNAL "")
add_subdirectory(eigen)
message("=================================")

message("Adding Submodule glog")
set(WITH_GFLAGS OFF CACHE INTERNAL "")
set(WITH_GTEST OFF CACHE INTERNAL "")
set(WITH_UNWIND OFF CACHE INTERNAL "")
add_subdirectory(glog)
SET(SAIGA_USE_GLOG 1 PARENT_SCOPE)
message("=================================")


message("=================================")
message("Adding Submodule ZLIB")


set(ZLIB_BUILD_SHARED OFF CACHE INTERNAL "")
set(ZLIB_BUILD_STATIC ON CACHE INTERNAL "")

# Need this because for some reason assimps exports zlib as well
set(SKIP_INSTALL_ALL ON CACHE INTERNAL "")
set(ZLIB_TARGET zlibstatic)
add_subdirectory(zlib)
set(SKIP_INSTALL_ALL OFF CACHE INTERNAL "")

# libPNG uses include_directory(ZLIB_INCLUDE_DIR) therefore we have to set these variables
set(ZLIB_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/zlib ${CMAKE_CURRENT_BINARY_DIR}/zlib CACHE PATH "zlib dir" FORCE)
set(ZLIB_INCLUDE_DIRS ${ZLIB_INCLUDE_DIR} CACHE PATH "zlib dir" FORCE)

# include dir that use the zlib target
target_include_directories(${ZLIB_TARGET} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/zlib>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/zlib>)

#set_target_properties(${ZLIB_TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${SAIGA_RUNTIME_OUTPUT_DIRECTORY}")
SET(SAIGA_USE_ZLIB 1 PARENT_SCOPE)
SET(ZLIB_FOUND 1)
set(ZLIB_LIBRARIES ${ZLIB_TARGET} CACHE INTERNAL "")


# Create a fake zlib target which also points to the zlib static library
add_library(zlib INTERFACE)
target_link_libraries(zlib INTERFACE zlibstatic)

message("=================================")


if (SAIGA_BUILD_GLFW)
    message("=================================")
    message("Adding Submodule glfw")
    set(BUILD_SHARED_LIBS ON CACHE INTERNAL "")
    set(GLFW_BUILD_EXAMPLES OFF CACHE INTERNAL "")
    set(GLFW_BUILD_TESTS OFF CACHE INTERNAL "")
    set(GLFW_BUILD_DOCS OFF CACHE INTERNAL "")

    add_subdirectory(${PROJECT_SOURCE_DIR}/submodules/glfw)
    set_target_properties(glfw PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${SAIGA_RUNTIME_OUTPUT_DIRECTORY}")
    SET(SAIGA_USE_GLFW 1 PARENT_SCOPE)

    #find_package(glfw3 CONFIG QUIET REQUIRED)
    SET(SAIGA_USE_GLFW 1)
#    PackageHelperTarget(glfw GLFW_FOUND)
    message("=================================")
endif ()

message("=================================")
message("Adding Submodule libPNG")

set(PNG_BUILD_ZLIB ON CACHE INTERNAL "")
set(PNG_STATIC ON CACHE INTERNAL "")
set(PNG_SHARED OFF CACHE INTERNAL "")
set(PNG_EXECUTABLES OFF CACHE INTERNAL "")
set(PNG_TESTS OFF CACHE INTERNAL "")

include_directories(${ZLIB_INCLUDE_DIRS})

add_subdirectory(libpng)

set_property(TARGET png_static PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(png_static PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/libpng>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/libpng>)

set_target_properties(png_static PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${SAIGA_RUNTIME_OUTPUT_DIRECTORY}")
SET(SAIGA_USE_PNG 1 PARENT_SCOPE)

#  set(CMAKE_INSTALL_LIBDIR lib)
#    install(TARGETS png zlib
#          EXPORT libpng
#          RUNTIME DESTINATION bin
#          LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#          ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
#          FRAMEWORK DESTINATION ${CMAKE_INSTALL_LIBDIR})
#
#  install(EXPORT libpng
#          DESTINATION lib/libpng)

message("=================================")


message("=================================")
message("Adding Submodule assimp")

set(ASSIMP_BUILD_TESTS OFF CACHE INTERNAL "")
set(ASSIMP_BUILD_ASSIMP_TOOLS OFF CACHE INTERNAL "")

add_subdirectory(assimp)
set_target_properties(assimp PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${SAIGA_RUNTIME_OUTPUT_DIRECTORY}")


SET(SAIGA_USE_ASSIMP 1 PARENT_SCOPE)
message("=================================")

message("=================================")
message("Adding Submodule googletest")
set(BUILD_SHARED_LIBS OFF CACHE INTERNAL "")
add_subdirectory(googletest)
set(BUILD_SHARED_LIBS ON CACHE INTERNAL "")
PackageHelperTarget(gtest GTEST_FOUND)
set(GTEST_FOUND 1 PARENT_SCOPE)
set(GTEST_TARGETS gtest gtest_main)
message("=================================")


