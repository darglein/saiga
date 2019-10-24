add_subdirectory(bow)
add_subdirectory(featureMatching)
add_subdirectory(eightPoint)
add_subdirectory(homography)
add_subdirectory(fivePoint)
add_subdirectory(robust_pose_optimization)
add_subdirectory(derive)
add_subdirectory(pnp)
add_subdirectory(registration)

if(OPENCV_FOUND)
add_subdirectory(orb)
endif()

if(SAIGA_USE_EIGENRECURSIVE)
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


    if (SDL2_FOUND AND SAIGA_USE_OPENGL)
        add_subdirectory(datasetViewer)
        if(G2O_FOUND)
            add_subdirectory(ba)
            add_subdirectory(posegraph_viewer)
        endif()
        if(OPENNI2_FOUND AND SAIGA_MODULE_EXTRA)
            add_subdirectory(openni2Viewer)
        endif()
    endif()
endif()
