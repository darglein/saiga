

FILE(GLOB_RECURSE TINY_GLOG_HEADER  "*.h" "*.hpp")
FILE(GLOB_RECURSE TINY_GLOG_SRC  "*.cpp")

option(TINY_GLOG_BUILD_SHARED "Build tiny-glog as a shared library" ON)

if (TINY_GLOG_BUILD_SHARED)
    add_library(tiny-glog SHARED ${TINY_GLOG_SRC})
else ()
    add_library(tiny-glog STATIC ${TINY_GLOG_SRC})
    set_property(TARGET tiny-glog PROPERTY POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(tiny-glog PUBLIC TINY_GLOG_STATIC)
endif ()

target_include_directories(tiny-glog INTERFACE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        )
add_library(glog::glog ALIAS tiny-glog)

