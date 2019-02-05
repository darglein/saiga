
if(SAIGA_MODULE_CORE)
add_subdirectory(core)
endif()

if(SAIGA_MODULE_VULKAN AND SAIGA_USE_VULKAN)
if (SDL2_FOUND AND SAIGA_USE_VULKAN)
    add_subdirectory(vulkan)
endif()
endif()


if(SAIGA_MODULE_OPENGL AND SAIGA_USE_OPENGL)
    add_subdirectory(opengl)
endif()


if(SAIGA_MODULE_CUDA AND SAIGA_USE_CUDA)
    add_subdirectory(cuda)
endif ()

if(SAIGA_MODULE_VISION AND SAIGA_VISION)
    add_subdirectory(vision)
endif()

if(SAIGA_MODULE_EXTRA)
add_subdirectory(modules)
endif()


return()










