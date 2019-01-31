# The module name is the directory name with "saiga_" as prefix
get_filename_component(MODULE_NAME ${CMAKE_CURRENT_LIST_DIR} NAME)
set(MODULE_NAME saiga_${MODULE_NAME})

# Collect all files in this directory
FILE(GLOB_RECURSE ${MODULE_NAME}_SRC  "*.cpp" "*.cu" "*.cc")
FILE(GLOB_RECURSE ${MODULE_NAME}_HEADER  "*.h" "*.hpp" "*.inl")
set(MODULE_ALL_FILES ${${MODULE_NAME}_SRC} ${${MODULE_NAME}_HEADER})

set(SAIGA_ALL_FILES ${SAIGA_ALL_FILES} ${MODULE_ALL_FILES} PARENT_SCOPE)



#add_subdirectory(assets)
#add_subdirectory(rendering)
#add_subdirectory(smaa)
#add_subdirectory(world)

#if(FREETYPE_FOUND AND SAIGA_USE_OPENGL)
#    add_subdirectory(text)
#endif()
