# Locates the SAIGA library




# Get the (current, i.e. installed) directory containing this file.
get_filename_component(SAIGA_CURRENT_CONFIG_DIR
"${CMAKE_CURRENT_LIST_FILE}" PATH)

if(NOT TARGET saiga::core)
include(${SAIGA_CURRENT_CONFIG_DIR}/SaigaTargets.cmake)
endif()

