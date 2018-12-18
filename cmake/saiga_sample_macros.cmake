macro(saiga_make_sample)

    # PREFIX: The directory name of the
    get_filename_component(PARENT_DIR ${CMAKE_CURRENT_LIST_DIR} DIRECTORY)
    get_filename_component(PREFIX ${PARENT_DIR} NAME)
    # Create target name from directory name
    get_filename_component(PROG_NAME ${CMAKE_CURRENT_LIST_DIR} NAME)
    string(REPLACE " " "_" PROG_NAME ${PROG_NAME})

    set(PROG_NAME "${PREFIX}_${PROG_NAME}")

    message(STATUS "Sample enabled:      ${PROG_NAME}")

    # Collect source and header files
    FILE(GLOB main_SRC  *.cpp)
    FILE(GLOB cuda_SRC  *.cu)
    FILE(GLOB main_HEADER  *.h)
    FILE(GLOB main_HEADER2  *.hpp)
    SET(PROG_SRC ${main_SRC} ${cuda_SRC} ${main_HEADER} ${main_HEADER2})

    include_directories(.)

    add_executable(${PROG_NAME} ${PROG_SRC} )
    target_link_libraries(${PROG_NAME} ${LIBS} ${LIB_NAME} )

    #set working directory for visual studio so the project can be executed from the ide
    set_target_properties(${PROG_NAME} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${OUTPUT_DIR}")
    set_target_properties(${PROG_NAME} PROPERTIES FOLDER samples/${PREFIX})
    set_target_properties(${PROG_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}")

endmacro()
