

if (GLFW_FOUND)
    add_subdirectory(simpleGLFWWindow)
endif ()

if (SDL2_FOUND)
    add_subdirectory(simpleSDLWindow)
    add_subdirectory(cornell)
    add_subdirectory(forwardRendering)
    add_subdirectory(splitScreen)
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
    if (OPENMESH_FOUND)
        add_subdirectory(depthMapTriangulation)
        add_subdirectory(openMeshSample)
    endif ()
    
    if(FREETYPE_FOUND)
        add_subdirectory(text_rendering)
    endif()

endif ()

if (EGL_FOUND)
    add_subdirectory(simpleEGLRendering)
endif ()

if(OPENVR_FOUND)
    add_subdirectory(hellovr)
endif()
