    add_subdirectory(ba_schur)
        add_subdirectory(block_matrices)

if (SDL2_FOUND AND SAIGA_USE_VULKAN)
    add_subdirectory(ba)
    if(OPENNI2_FOUND)
        add_subdirectory(openni2Viewer)
    endif()
endif()

if (OPENNI2_FOUND AND Boost_FOUND)
    add_subdirectory(openni2Client)
    add_subdirectory(openni2Server)
endif ()
