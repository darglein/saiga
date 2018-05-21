set(PROG_NAME "simpleBullet")

FILE(GLOB main_SRC  *.cpp)

SET(PROG_SRC ${main_SRC})

include_directories(.)


add_executable(${PROG_NAME} ${PROG_SRC})
target_link_libraries(${PROG_NAME} ${LIBS} ${LIB_NAME} )

if(${CMAKE_VERSION} VERSION_GREATER "3.8.0") 
    #set working directory for visual studio so the project can be executed from the ide
	set_target_properties(${PROG_NAME} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
endif()


set_target_properties (${PROG_NAME} PROPERTIES FOLDER samples)
	
set_target_properties( ${PROG_NAME}
	PROPERTIES
    	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
