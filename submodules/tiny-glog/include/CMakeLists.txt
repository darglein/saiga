

FILE(GLOB_RECURSE TINY_GLOG_HEADER  "*.h" "*.hpp")
FILE(GLOB_RECURSE TINY_GLOG_SRC  "*.cpp")

add_library(tiny-glog SHARED ${TINY_GLOG_SRC})

target_include_directories(tiny-glog INTERFACE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        )
add_library(glog::glog ALIAS tiny-glog)

