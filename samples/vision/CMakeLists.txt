add_subdirectory(bow)
add_subdirectory(featureMatching_benchmark)
add_subdirectory(robust_pose_optimization)
add_subdirectory(derive)

if(EIGENRECURSIVE_FOUND)
    #    add_subdirectory(block_matrices)

    if(SAIGA_USE_CHOLMOD)
        add_subdirectory(sparse_ldlt)
    endif()


    if(G2O_FOUND)
        add_subdirectory(posegraph)
        add_subdirectory(ba_performance_test)
    endif()



    if(SAIGA_MODULE_EXTRA AND OPENMESH_FOUND)
        add_subdirectory(arap)
    endif()

    if (SDL2_FOUND AND SAIGA_USE_VULKAN)
        if(G2O_FOUND)
            add_subdirectory(ba)
            add_subdirectory(posegraph_viewer)
        endif()
        if(OPENNI2_FOUND AND SAIGA_MODULE_EXTRA)
            add_subdirectory(openni2Viewer)
        endif()
    endif()
endif()
