
if(MODULE_CORE)
    add_subdirectory(core)
endif()


if(MODULE_OPENGL)
    add_subdirectory(opengl)
endif()


if(MODULE_CUDA)
    add_subdirectory(cuda)
endif ()

if(MODULE_EXTRA)
    add_subdirectory(modules)
endif()


if(MODULE_VISION)
    add_subdirectory(vision)
endif()

if(MODULE_VULKAN)
    # All vulkan samples also need sdl2
    if (SDL2_FOUND)
        add_subdirectory(vulkan)
    endif()
endif()












