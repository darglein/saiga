set(PROG_NAME "eigenTest")

FILE(GLOB main_SRC  *.cpp)

SET(PROG_SRC ${main_SRC})

include_directories(.)

#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

add_executable(${PROG_NAME} ${PROG_SRC})
target_link_libraries(${PROG_NAME} ${LIBS} ${LIB_NAME} )


set_target_properties( ${PROG_NAME}
	PROPERTIES
    	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
