# Games built on saiga_engine: one library per game in games/<name>/ (saiga_game_<name>), the executables are in
# samples/vulkan/ (docs/engine_design.md §5.3, §18.1). Built when MODULE_ENGINE is set, independent of
# SAIGA_BUILD_SAMPLES.
#
# The game libraries are static (no export macros; only the executables and the game tests link them). They include
# their headers with the path from the repository root: #include "games/<name>/<File>.h".

function(saiga_game_library NAME)
    set(TARGET_NAME saiga_game_${NAME})
    file(GLOB GAME_SRC "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
    file(GLOB GAME_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
    add_library(${TARGET_NAME} STATIC ${GAME_SRC} ${GAME_HEADER})
    target_link_libraries(${TARGET_NAME} PUBLIC saiga_engine ${ARGN})
    target_include_directories(${TARGET_NAME} PUBLIC ${PROJECT_SOURCE_DIR})
    target_compile_options(${TARGET_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${SAIGA_PRIVATE_CXX_FLAGS}>")
    set_target_properties(${TARGET_NAME} PROPERTIES FOLDER games)
    message(STATUS "Game enabled:        ${NAME}")
endfunction()

# Tech demos (§18.4)
add_subdirectory(common)
add_subdirectory(flight)
add_subdirectory(platformer)
add_subdirectory(racing)
