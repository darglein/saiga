message(" ")

if(MODULE_CORE)
    add_subdirectory(core)
    message(STATUS "Module enabled:     core")
endif()

if(MODULE_CUDA)
    message(STATUS "Module enabled:     cuda")
    add_subdirectory(cuda)
endif()

if(MODULE_EXTRA)
    message(STATUS "Module enabled:     extra")
    add_subdirectory(extra)
endif()

if(MODULE_OPENGL)
    message(STATUS "Module enabled:     opengl")
    add_subdirectory(opengl)
endif()

if(MODULE_VULKAN)
    message(STATUS "Module enabled:     vulkan")
    add_subdirectory(vulkan)
endif()

if(MODULE_VISION)
    message(STATUS "Module enabled:     vision")
    add_subdirectory(vision)
endif()



install(
    EXPORT ${LIB_EXPORT}
    FILE SaigaTargets.cmake
    NAMESPACE Saiga::
    DESTINATION ${INSTALL_SHARE}
    )
