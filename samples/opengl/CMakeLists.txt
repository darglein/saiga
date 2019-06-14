

if (GLFW_FOUND)
    add_subdirectory(simpleGLFWWindow)
endif ()

if (SDL2_FOUND)
    add_subdirectory(simpleSDLWindow)
    add_subdirectory(multiWindow)
    add_subdirectory(forwardRendering)
    add_subdirectory(cascadedShadowMaps)
    add_subdirectory(lighting)
    add_subdirectory(volumetricLights)
    add_subdirectory(modelViewer)
    if (FFMPEG_FOUND)
        add_subdirectory(videoRecording)
    endif ()

    if (BULLET_FOUND)
        add_subdirectory(simpleBullet)
    endif ()
    if(FREETYPE_FOUND)
        add_subdirectory(stopwatch)
    endif()
    if (OPENMESH_FOUND)
        add_subdirectory(openMeshSample)
    endif ()

endif ()

if (EGL_FOUND)
    add_subdirectory(offscreenRendering)
endif ()

