cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(TinyGlog LANGUAGES CXX)
add_subdirectory("include")


#install(TARGETS tiny-glog
#        EXPORT tiny-glog-targets
#        LIBRARY DESTINATION lib COMPONENT Runtime
#        ARCHIVE DESTINATION lib COMPONENT Development
#        RUNTIME DESTINATION bin COMPONENT Runtime
#        PUBLIC_HEADER DESTINATION include COMPONENT Development
#        BUNDLE DESTINATION bin COMPONENT Runtime
#        )
#
#install(EXPORT tiny-glog-targets DESTINATION lib/cmake/mylib)