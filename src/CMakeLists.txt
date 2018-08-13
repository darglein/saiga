

add_subdirectory(saiga)
add_subdirectory(internal)

#creates 'folders' in the solution explorer of visual studio
#GroupSources2(${PROJECT_SOURCE_DIR}/src)
#GroupSources2(${PROJECT_SOURCE_DIR}/include/saiga)
#GroupSources(${PROJECT_SOURCE_DIR} shader)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

SET(SAIGA_ALL_FILES
    ${SAIGA_SOURCE_CORE}
    ${SAIGA_SOURCE_EXT}
    ${SAIGA_HEADERS}
    ${SAIGA_SHADERS}
    ${SAIGA_INTERNAL_SOURCE}
    ${SAIGA_INTERNAL_HEADERS}
    )

if(SAIGA_BUILD_SHARED)
    add_definitions(-DSAIGA_DLL_EXPORTS)
    add_library(${LIB_NAME} SHARED ${SAIGA_ALL_FILES})
else ()
    add_library(${LIB_NAME} STATIC ${SAIGA_ALL_FILES})
endif()


target_link_libraries(${LIB_NAME} ${LIBS} ${ALL_CUDA_LIBS})

set_target_properties( ${LIB_NAME}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")


install(
    TARGETS ${LIB_NAME}
    RUNTIME DESTINATION ${INSTALL_BIN}
    LIBRARY DESTINATION ${INSTALL_SHARED}
    ARCHIVE DESTINATION ${INSTALL_LIB}
    )

install(
    DIRECTORY ./saiga/
    DESTINATION ${INSTALL_INCLUDE}
    FILES_MATCHING PATTERN "*.h"
    PATTERN "*.inl"
    )

install(
    DIRECTORY ./saiga/
    DESTINATION ${INSTALL_INCLUDE}
    FILES_MATCHING PATTERN "*.hpp"
    PATTERN "*.inl"
    )




