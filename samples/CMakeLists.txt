macro(saiga_make_sample _modules)
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

    # We only need to link the saiga target
    target_link_libraries(${PROG_NAME} ${${_modules}})

    #set working directory for visual studio so the project can be executed from the ide
    set_target_properties(${PROG_NAME} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${OUTPUT_DIR}")
    set_target_properties(${PROG_NAME} PROPERTIES FOLDER samples/${PREFIX})
    #set_target_properties(${PROG_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
endmacro()

macro(saiga_make_benchmark_sample)
    # Make sure this always compiles with optimizations
    set(CMAKE_BUILD_TYPE Release)

    if(SAIGA_CXX_CLANG OR SAIGA_CXX_GNU)
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    endif()
    if(SAIGA_CXX_MSVC)
        #todo check if avx is present
        # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Oi /Ot /Oy /GL /fp:fast /Gy /arch:AVX2")
        # set(CMAKE_LD_FLAGS "${CMAKE_LD_FLAGS} /LTCG")
        # add_definitions(-D__FMA__)
    endif()
endmacro()


if(MODULE_CORE)
    add_subdirectory(core)
endif()


if(MODULE_OPENGL)
    add_subdirectory(opengl)
endif()


if(MODULE_CUDA)
    add_subdirectory(cuda)
endif ()


if(MODULE_VISION)
    add_subdirectory(vision)
endif()

if(MODULE_VULKAN)
    # All vulkan samples also need sdl2
    if (SDL2_FOUND)
        add_subdirectory(vulkan)
    endif()
endif()












