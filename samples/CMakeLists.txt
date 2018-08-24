
if(CUDA_FOUND)
    add_subdirectory(cuda)
endif ()

if(OPENGL_FOUND)
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
        if (FFMPEG_FOUND)
            add_subdirectory(videoRecording)
        endif ()

        if (BULLET_FOUND)
            add_subdirectory(simpleBullet)
        endif ()
        if(FREETYPE_FOUND)
            add_subdirectory(advancedSDLWindow)
            add_subdirectory(stopwatch)
        endif()
        if (OPENMESH_FOUND)
            add_subdirectory(openMeshSample)
        endif ()

    endif ()

    if (EGL_FOUND)
        add_subdirectory(offscreenRendering)
    endif ()


    if (EIGEN3_FOUND)
        add_subdirectory(eigenTest)
    endif ()


    add_subdirectory(imageProcessing)

endif()
if (SDL2_FOUND AND SAIGA_USE_VULKAN)
    add_subdirectory(vulkan)
endif()

if (OPENNI2_FOUND AND Boost_FOUND)
    add_subdirectory(openni2Client)
    add_subdirectory(openni2Server)
endif ()
