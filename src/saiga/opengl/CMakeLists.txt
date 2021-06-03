set(MODULE_NAME "saiga_opengl")


add_subdirectory(animation)
add_subdirectory(assets)

if(OPENVR_FOUND)
    add_subdirectory(vr)
endif()


if(ASSIMP_FOUND)
    add_subdirectory(assimp)
endif()

if(EGL_FOUND)
    add_subdirectory(egl)
endif()

if(FFMPEG_FOUND)
    add_subdirectory(ffmpeg)
endif()

add_subdirectory(glbinding)
add_subdirectory(imgui)
add_subdirectory(query)
add_subdirectory(rendering)

add_subdirectory(shader)
add_subdirectory(smaa)

if(FREETYPE_FOUND)
    add_subdirectory(text)
endif()

add_subdirectory(texture)
add_subdirectory(window)
add_subdirectory(world)




FILE(GLOB OPENGL_SRC  "*.cpp" "*.cu" "*.cc")
FILE(GLOB OPENGL_HEADER  "*.h" "*.hpp" "*.inl" "Opengl")


# the internal files are added to core
SET(SAIGA_ALL_FILES
    ${SAIGA_ALL_FILES}
    ${OPENGL_SRC}
    ${OPENGL_HEADER}
    )


if(SAIGA_BUILD_SHARED)
    add_definitions(-DSAIGA_DLL_EXPORTS)
    add_library(${MODULE_NAME} SHARED ${SAIGA_ALL_FILES})
else ()
    add_library(${MODULE_NAME} STATIC ${SAIGA_ALL_FILES})
endif()

target_link_libraries(${MODULE_NAME} ${OPENGL_TARGETS} ${OPENGL_LIBS})
target_compile_options(${MODULE_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${SAIGA_PRIVATE_CXX_FLAGS}>")

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
    "$<BUILD_INTERFACE:${OPENGL_INCLUDES}>"
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

if(SAIGA_PCH)
    target_precompile_headers(
        ${MODULE_NAME}
      PRIVATE
        "${PROJECT_SOURCE_DIR}/src/internal/pch_opengl.h"
    )
endif()
