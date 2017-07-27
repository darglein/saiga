set(PROG_NAME "twoViewOrientation")

FILE(GLOB main_SRC  *.cpp)

SET(PROG_SRC ${main_SRC})

include_directories(.)


add_executable(${PROG_NAME} ${PROG_SRC})
target_link_libraries(${PROG_NAME} ${LIBS} ${LIB_NAME} )


set_target_properties( ${PROG_NAME}
	PROPERTIES
    	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
