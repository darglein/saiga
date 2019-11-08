set(MODULE_NAME "saiga_cuda")




##############################################################################################################################################################



FILE(GLOB_RECURSE VULKAN_SRC  "*.cpp" "*.cu" "*.cc")
FILE(GLOB_RECURSE VULKAN_HEADER  "*.h" "*.hpp" "*.inl")


# the internal files are added to core
SET(SAIGA_ALL_FILES
    ${VULKAN_SRC}
    ${VULKAN_HEADER}
    )


if(SAIGA_BUILD_SHARED)
    add_definitions(-DSAIGA_DLL_EXPORTS)
    add_library(${MODULE_NAME} SHARED ${SAIGA_ALL_FILES})
else ()
    add_library(${MODULE_NAME} STATIC ${SAIGA_ALL_FILES})
endif()

target_link_libraries(${MODULE_NAME}  saiga_core )
target_link_libraries(${MODULE_NAME} ${ALL_CUDA_LIBS}  )
target_link_libraries(${MODULE_NAME}  ${LIBS} )
target_link_libraries(${MODULE_NAME} ${LIB_TARGETS} )

target_include_directories(
    ${MODULE_NAME}
    PUBLIC
    $<INSTALL_INTERFACE:include>
    $<INSTALL_INTERFACE:include/saiga/opengl/glbinding/include>
    $<INSTALL_INTERFACE:share/saiga/shader/include>
    #"$<INSTALL_INTERFACE:${PACKAGE_INCLUDES}>"
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/saiga/opengl/glbinding/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/shader/include>
    "$<BUILD_INTERFACE:${PACKAGE_INCLUDES}>"
    #PRIVATE
    # for the internal files
    #${PROJECT_SOURCE_DIR}/src/
    )

install(
    TARGETS ${MODULE_NAME}
    EXPORT ${LIB_EXPORT}
    RUNTIME DESTINATION ${INSTALL_BIN}
    LIBRARY DESTINATION ${INSTALL_SHARED}
    ARCHIVE DESTINATION ${INSTALL_LIB}
    )

