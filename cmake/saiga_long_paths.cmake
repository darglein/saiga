# A cache variable, because the function can be called from projects that include saiga with add_subdirectory and
# don't see the variables of its directory.
set(SAIGA_LONG_PATHS_MANIFEST "${CMAKE_CURRENT_LIST_DIR}/saiga_long_paths.manifest" CACHE INTERNAL "")

# Makes an executable long path aware on Windows (longPathAware in its manifest): on systems that enable long paths
# (LongPathsEnabled in the registry), its file functions then take paths longer than MAX_PATH without the \\?\ prefix,
# also in libraries that don't use Saiga::os_path. Only MSVC and clang-cl builds get the manifest; on other platforms
# and toolchains this does nothing, and Saiga::os_path alone handles long paths.
function(saiga_enable_long_paths TARGET_NAME)
    get_target_property(target_type ${TARGET_NAME} TYPE)
    if (NOT target_type STREQUAL "EXECUTABLE")
        message(FATAL_ERROR "saiga_enable_long_paths: ${TARGET_NAME} is not an executable")
    endif ()
    if (NOT MSVC)
        return()
    endif ()
    # The linker merges it with the manifest it generates.
    target_sources(${TARGET_NAME} PRIVATE "${SAIGA_LONG_PATHS_MANIFEST}")
endfunction()
