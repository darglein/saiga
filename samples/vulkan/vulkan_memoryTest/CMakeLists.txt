# Create target name from directory name
get_filename_component(PROG_NAME ${CMAKE_CURRENT_LIST_DIR} NAME)
string(REPLACE " " "_" PROG_NAME ${PROG_NAME})

# Collect source and header files
FILE(GLOB main_SRC  *.cpp)
FILE(GLOB_RECURSE main_HEADER  *.h)
FILE(GLOB_RECURSE main_HEADER2  *.hpp)
SET(PROG_SRC ${main_SRC} ${main_HEADER} ${main_HEADER2})

include_directories(.)
include_directories(base)

add_executable(${PROG_NAME} ${PROG_SRC} )
target_link_libraries(${PROG_NAME} ${LIBS} ${LIB_NAME} )
set_property(TARGET ${PROG_NAME} PROPERTY CXX_STANDARD 17)
if(${CMAKE_VERSION} VERSION_GREATER "3.8.0") 
    #set working directory for visual studio so the project can be executed from the ide
	set_target_properties(${PROG_NAME} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${OUTPUT_DIR}")
endif()

set_target_properties (${PROG_NAME} PROPERTIES FOLDER samples)
set_target_properties( ${PROG_NAME}
	PROPERTIES
    	RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}/vulkan")
