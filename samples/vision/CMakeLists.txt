
if (SDL2_FOUND AND SAIGA_USE_VULKAN)
    add_subdirectory(vision_ba)
    if(OPENNI2_FOUND)
        add_subdirectory(vision_openni2Viewer)
    endif()
endif()
