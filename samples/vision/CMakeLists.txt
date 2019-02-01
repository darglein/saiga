add_subdirectory(ba_schur)
add_subdirectory(block_matrices)


if(G2O_FOUND)
    add_subdirectory(posegraph)
    add_subdirectory(ba_performance_test)
endif()

if(SAIGA_USE_MKL)
    add_subdirectory(mkl_test)
endif()

if (SDL2_FOUND AND SAIGA_USE_VULKAN)
    if(G2O_FOUND)
        add_subdirectory(ba)
        add_subdirectory(posegraph_viewer)
    endif()
    if(OPENNI2_FOUND)
        add_subdirectory(openni2Viewer)
    endif()
endif()

#if (OPENNI2_FOUND AND Boost_FOUND)
#    add_subdirectory(openni2Client)
#    add_subdirectory(openni2Server)
#endif ()
