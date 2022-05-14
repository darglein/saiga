cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(TinyEigen LANGUAGES CXX)
add_subdirectory("include")


install(TARGETS tiny-eigen
        EXPORT tiny-eigen-targets
        LIBRARY DESTINATION lib COMPONENT Runtime
        ARCHIVE DESTINATION lib COMPONENT Development
        RUNTIME DESTINATION bin COMPONENT Runtime
        PUBLIC_HEADER DESTINATION include COMPONENT Development
        BUNDLE DESTINATION bin COMPONENT Runtime
        )

install(EXPORT tiny-eigen-targets DESTINATION lib/cmake/mylib)