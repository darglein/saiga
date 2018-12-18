
if (SDL2_FOUND AND SAIGA_USE_VULKAN)
    add_subdirectory(ba)
    if(OPENNI2_FOUND)
        add_subdirectory(openni2Viewer)
    endif()
endif()
