# The module name is the directory name with "saiga_" as prefix
get_filename_component(MODULE_NAME ${CMAKE_CURRENT_LIST_DIR} NAME)
set(MODULE_NAME saiga_${MODULE_NAME})

# Collect all files in this directory
FILE(GLOB ${MODULE_NAME}_SRC  "*.cpp" )
FILE(GLOB ${MODULE_NAME}_HEADER  "*.h" "*.inl")

FILE(GLOB ${MODULE_NAME}_ICP_SRC  "icp/*.cpp" )
FILE(GLOB ${MODULE_NAME}_ICP_HEADER  "icp/*.h")

FILE(GLOB ${MODULE_NAME}_KERNELS_HEADER  "kernels/*.h")

FILE(GLOB ${MODULE_NAME}_RECURSIVE_HEADER  "recursiveMatrices/*.h")

FILE(GLOB ${MODULE_NAME}_BA  "ba/*")

set(MODULE_ALL_FILES
    ${${MODULE_NAME}_SRC}
    ${${MODULE_NAME}_HEADER}
    ${${MODULE_NAME}_ICP_SRC}
    ${${MODULE_NAME}_ICP_HEADER}
    ${${MODULE_NAME}_KERNELS_HEADER}
    ${${MODULE_NAME}_RECURSIVE_HEADER}
    ${${MODULE_NAME}_BA}
    )

if(G2O_FOUND)
    FILE(GLOB_RECURSE ${MODULE_NAME}_G2O_SRC  "g2o/*.cpp" )
    FILE(GLOB_RECURSE ${MODULE_NAME}_G2O_HEADER  "g2o/*.h")
    set(MODULE_ALL_FILES
        ${MODULE_ALL_FILES}
        ${${MODULE_NAME}_G2O_SRC}
        ${${MODULE_NAME}_G2O_HEADER}
        )
endif()

if(CERES_FOUND)
    FILE(GLOB_RECURSE ${MODULE_NAME}_CERES_SRC  "ceres/*.cpp" )
    FILE(GLOB_RECURSE ${MODULE_NAME}_CERES_HEADER  "ceres/*.h")
    set(MODULE_ALL_FILES
        ${MODULE_ALL_FILES}
        ${${MODULE_NAME}_CERES_SRC}
        ${${MODULE_NAME}_CERES_HEADER}
        )
endif()


set(SAIGA_ALL_FILES ${SAIGA_ALL_FILES} ${MODULE_ALL_FILES} PARENT_SCOPE)
