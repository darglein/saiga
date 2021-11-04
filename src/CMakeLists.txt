add_subdirectory(internal)
add_subdirectory(saiga)

# This test executable is always build and will be installed into (INSTALL_PREFIX)/bin
add_executable(saiga_test saiga_test.cpp)
target_link_libraries(saiga_test saiga_core)
set_target_properties(saiga_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
install(
    TARGETS saiga_test
    RUNTIME DESTINATION ${INSTALL_BIN}
    )
