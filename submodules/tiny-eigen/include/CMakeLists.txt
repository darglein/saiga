

FILE(GLOB_RECURSE TINY_EIGEN_HEADER  "*.h" "*.hpp")

add_library(tiny-eigen INTERFACE)
#target_sources(tiny-eigen INTERFACE ${TINY_EIGEN_HEADER})
target_include_directories(tiny-eigen INTERFACE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        )
add_library(Eigen3::Eigen ALIAS tiny-eigen)

