include(saiga_sample_macros)
project(vision_orb)
find_package(Pangolin QUIET REQUIRED)
find_package(OpenCV QUIET REQUIRED)
find_package(Eigen3 QUIET REQUIRED)
include_directories(.)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -march=native -fopenmp")

add_executable(vision_orb orb_sample.cpp orb_sample.h ../../../src/saiga/vision/orb/ORBextractor.h
        ../../../src/saiga/vision/orb/ORBextractor.cpp ../../../src/saiga/vision/orb/Distribution.h
        ../../../src/saiga/vision/orb/Distribution.cpp ../../../src/saiga/vision/orb/FAST.h
        ../../../src/saiga/vision/orb/Fast.cpp ../../../src/saiga/vision/orb/GaussianBlur.h
        ../../../src/saiga/vision/orb/ORB_config.h ../../../src/saiga/vision/orb/ORBconstants.h
        ../../../src/saiga/vision/orb/Types.h)

target_link_libraries(vision_orb ${OpenCV_LIBS} ${Pangolin_LIBRARIES} saiga_vision saiga_extra)


#saiga_make_sample(required_modules)