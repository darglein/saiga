
add_subdirectory(core)

if (SDL2_FOUND AND SAIGA_USE_VULKAN)
    add_subdirectory(vulkan)
endif()
if(SAIGA_USE_OPENGL)
    add_subdirectory(opengl)
endif()


if(CUDA_FOUND)
    add_subdirectory(cuda)
endif ()
if(SAIGA_VISION)
    add_subdirectory(vision)
endif()

add_subdirectory(modules)

return()










