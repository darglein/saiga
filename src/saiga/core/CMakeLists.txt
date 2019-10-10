set(MODULE_NAME "saiga_core")

add_subdirectory(camera)
add_subdirectory(framework)
add_subdirectory(math)
add_subdirectory(model)
add_subdirectory(window)
add_subdirectory(image)
add_subdirectory(time)
add_subdirectory(geometry)
add_subdirectory(util)
add_subdirectory(tests)
add_subdirectory(imgui)
add_subdirectory(rendering)

if (SDL2_FOUND)
    add_subdirectory(sdl)
endif ()

if (GLFW_FOUND)
    add_subdirectory(glfw)
endif ()

if (OPENAL_FOUND)
    add_subdirectory(sound)
endif ()

# the internal files are added to core
SET(SAIGA_ALL_FILES
    Core
    ${SAIGA_ALL_FILES}
    ${SAIGA_INTERNAL_SOURCE}
    ${SAIGA_INTERNAL_HEADERS}
    )

include_directories(${PROJECT_SOURCE_DIR}/src/)

if(SAIGA_BUILD_SHARED)
    add_definitions(-DSAIGA_DLL_EXPORTS)
    add_library(${MODULE_NAME} SHARED ${SAIGA_ALL_FILES})
else ()
    add_library(${MODULE_NAME} STATIC ${SAIGA_ALL_FILES})
endif()


target_link_libraries(${MODULE_NAME} PUBLIC ${CORE_LIBS} ${CORE_TARGETS} )
target_link_libraries(${MODULE_NAME} PRIVATE std::filesystem)

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
    "$<BUILD_INTERFACE:${CORE_INCLUDES}>"
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

#install(
#    EXPORT ${LIB_EXPORT}
#    FILE SaigaTargets.cmake
#    DESTINATION ${INSTALL_SHARE}
#    )

##### Precompiled Headers #####
if(SAIGA_PCH)
    set_target_properties(${MODULE_NAME} PROPERTIES COTIRE_CXX_PREFIX_HEADER_INIT "${PROJECT_SOURCE_DIR}/src/internal/pch_core.h")
    set_target_properties(${MODULE_NAME} PROPERTIES COTIRE_ADD_UNITY_BUILD false)
    cotire(${MODULE_NAME})
endif()
