

add_subdirectory(saiga)
add_subdirectory(internal)

#creates 'folders' in the solution explorer of visual studio
GroupSources2(${PROJECT_SOURCE_DIR}/src/saiga)
GroupSources2(${PROJECT_SOURCE_DIR}/src/internal)
GroupSources(${PROJECT_SOURCE_DIR} shader)


SET(SAIGA_ALL_FILES
    ${SAIGA_ALL_FILES}
    ${SAIGA_INTERNAL_SOURCE}
    ${SAIGA_INTERNAL_HEADERS}
    )

if(SAIGA_BUILD_SHARED)
    add_definitions(-DSAIGA_DLL_EXPORTS)
    add_library(${LIB_NAME} SHARED ${SAIGA_ALL_FILES})
else ()
    add_library(${LIB_NAME} STATIC ${SAIGA_ALL_FILES})
endif()



#message(STATUS "libs: ${LIBS}")

target_link_libraries(${LIB_NAME} ${ALL_CUDA_LIBS} ${LIBS} ${LIB_TARGETS} )


target_include_directories(
    ${LIB_NAME}
    PUBLIC
    $<INSTALL_INTERFACE:include>
    $<INSTALL_INTERFACE:include/saiga/opengl/glbinding/include>
    $<INSTALL_INTERFACE:include/saiga/opengl/glbinding-aux/include>
    $<INSTALL_INTERFACE:share/saiga/shader/include>
    #"$<INSTALL_INTERFACE:${PACKAGE_INCLUDES}>"
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/saiga/opengl/glbinding/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/saiga/opengl/glbinding-aux/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/shader/include>
    "$<BUILD_INTERFACE:${PACKAGE_INCLUDES}>"
    #PRIVATE
    # for the internal files
    #${PROJECT_SOURCE_DIR}/src/
    )


install(
    TARGETS ${LIB_NAME}
    EXPORT ${LIB_EXPORT}
    RUNTIME DESTINATION ${INSTALL_BIN}
    LIBRARY DESTINATION ${INSTALL_SHARED}
    ARCHIVE DESTINATION ${INSTALL_LIB}
    )

install(
    EXPORT ${LIB_EXPORT}
    FILE SaigaTargets.cmake
    DESTINATION ${INSTALL_SHARE}
    )


install(
    DIRECTORY ./saiga/
    DESTINATION ${INSTALL_INCLUDE}
    FILES_MATCHING PATTERN "*.h"
    PATTERN "*.inl"
    PATTERN "*.hpp"
    )




