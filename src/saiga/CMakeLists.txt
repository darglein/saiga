message(" ")

if(MODULE_CORE)
    add_subdirectory(core)
    message(STATUS "Module enabled:     Core")
endif()

if(MODULE_OPENGL)
    message(STATUS "Module enabled:     Opengl")
    add_subdirectory(opengl)
endif()

if(MODULE_VULKAN)
    message(STATUS "Module enabled:     Vulkan")
    add_subdirectory(vulkan)
endif()

if(MODULE_VISION)
    message(STATUS "Module enabled:     Vision")
    add_subdirectory(vision)
endif()

if(MODULE_CUDA)
    message(STATUS "Module enabled:     Cuda")
    add_subdirectory(cuda)
endif()


install(
    EXPORT ${LIB_EXPORT}
    FILE SaigaTargets.cmake
    NAMESPACE Saiga::
    DESTINATION ${INSTALL_SHARE}

    )
