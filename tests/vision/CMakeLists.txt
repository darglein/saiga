add_subdirectory(motionModel)
add_subdirectory(PoseEstimation)

if(CERES_FOUND)
    add_subdirectory(BundleAdjustment)
endif()


