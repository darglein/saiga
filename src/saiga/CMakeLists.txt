message(" ")

if(SAIGA_MODULE_CORE)
    add_subdirectory(core)
    message(STATUS "Module enabled:     core")
endif()

if(SAIGA_MODULE_VULKAN AND SAIGA_USE_VULKAN)
    message(STATUS "Module enabled:     vulkan")
    add_subdirectory(vulkan)
endif()

if(SAIGA_MODULE_OPENGL AND SAIGA_USE_OPENGL)
    message(STATUS "Module enabled:     opengl")
    add_subdirectory(opengl)
endif()

if(SAIGA_MODULE_EXTRA)
    message(STATUS "Module enabled:     extra")
    add_subdirectory(extra)
endif()

if(SAIGA_MODULE_CUDA AND SAIGA_USE_CUDA)
    message(STATUS "Module enabled:     cuda")
    add_subdirectory(cuda)
endif()

if(SAIGA_MODULE_VISION AND SAIGA_VISION)
    message(STATUS "Module enabled:     vision")
    add_subdirectory(vision)
endif()
