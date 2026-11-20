# Tech demo games on saiga_engine (docs/engine_design.md §18.4): one executable game_<name> per samples/games/<name>/
# (its main.cpp and the other sources of the directory and its subdirectories, e.g. content/), plus the shared code of samples/games/common/, which is
# compiled into each game. The sources include the headers as "games/<name>/<File>.h" (include directory samples/).

function(saiga_game NAME)
    set(TARGET_NAME game_${NAME})
    file(GLOB_RECURSE GAME_SRC "${CMAKE_CURRENT_SOURCE_DIR}/${NAME}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/${NAME}/*.h")
    file(GLOB COMMON_SRC "${CMAKE_CURRENT_SOURCE_DIR}/common/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/common/*.h")
    add_executable(${TARGET_NAME} ${GAME_SRC} ${COMMON_SRC})
    target_link_libraries(${TARGET_NAME} PRIVATE saiga_core saiga_vulkan saiga_engine)
    target_include_directories(${TARGET_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/samples)
    target_compile_options(${TARGET_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${SAIGA_PRIVATE_CXX_FLAGS}>")
    set_target_properties(${TARGET_NAME} PROPERTIES FOLDER samples/games)
    message(STATUS "Sample enabled:      ${TARGET_NAME}")
endfunction()

saiga_game(flight)
saiga_game(platformer)
saiga_game(racing)
saiga_game(world)
