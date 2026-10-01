set(SAIGA_VERSION_RESOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}")
if (MSVC)
    enable_language(RC)
endif ()

# Adds a Windows VERSIONINFO resource (version, commit, copyright) to a shared saiga module, shown in the DLL's file
# properties. Uses SAIGA_VERSION and SAIGA_GIT_SHA1 from the top-level CMakeLists.txt.
function(saiga_add_version_resource TARGET_NAME FILE_DESCRIPTION)
    get_target_property(target_type ${TARGET_NAME} TYPE)
    if (NOT MSVC OR NOT target_type STREQUAL "SHARED_LIBRARY")
        return()
    endif ()

    set(VERSION_STRING "${SAIGA_VERSION}")
    if (SAIGA_GIT_SHA1 MATCHES "^[0-9a-f]+$")
        string(SUBSTRING "${SAIGA_GIT_SHA1}" 0 7 short_sha)
        string(APPEND VERSION_STRING " (${short_sha})")
    endif ()
    set(VERSION_FILE_DESCRIPTION "${FILE_DESCRIPTION}")

    set(VERSION_RC "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}_version.rc")
    configure_file("${SAIGA_VERSION_RESOURCE_DIR}/saiga_version.rc.in" "${VERSION_RC}" @ONLY)
    target_sources(${TARGET_NAME} PRIVATE "${VERSION_RC}")
endfunction()
