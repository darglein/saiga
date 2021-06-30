macro(saiga_vision_sample FILE_NAME)
  string(REGEX REPLACE "\\.[^.]*$" "" NAME ${FILE_NAME})

  set(TARGET_NAME ${NAME})

  add_executable(${TARGET_NAME} ${FILE_NAME})
  message(STATUS "Sample enabled:      ${NAME}")


  target_link_libraries(${TARGET_NAME} PUBLIC saiga_core saiga_vision)
  target_link_libraries(${TARGET_NAME} PUBLIC ${ARGN})

  set_target_properties(${TARGET_NAME} PROPERTIES FOLDER samples/${PREFIX})
endmacro()

saiga_vision_sample(sample_vision_calib_response.cpp)
saiga_vision_sample(sample_vision_bow.cpp)
saiga_vision_sample(sample_vision_derive.cpp)
saiga_vision_sample(sample_vision_featureMatching.cpp)
saiga_vision_sample(sample_vision_fivePoint.cpp)

saiga_vision_sample(sample_vision_homography.cpp)
saiga_vision_sample(sample_vision_pnp.cpp)
saiga_vision_sample(sample_vision_registration.cpp)
saiga_vision_sample(sample_vision_robust_pose_optimization.cpp)

if(SAIGA_USE_CHOLMOD)
  saiga_vision_sample(sample_vision_sparse_ldlt.cpp)
endif()


if(G2O_FOUND AND CERES_FOUND)
  saiga_vision_sample(sample_vision_posegraph.cpp)
  saiga_vision_sample(sample_vision_ba_benchmark.cpp)
endif()

#gphoto2
find_package(GPHOTO2 REQUIRED)
if(GPHOTO2_FOUND)
  saiga_vision_sample(sample_vision_hdr_capture.cpp ${GPHOTO2_LIBRARIES})
  target_link_libraries(sample_vision_hdr_capture PUBLIC ${GPHOTO2_LIBRARIES})
  target_include_directories(sample_vision_hdr_capture PUBLIC ${GPHOTO2_INCLUDE_DIRS})
endif()

if(SAIGA_MODULE_EXTRA AND OPENMESH_FOUND AND CERES_FOUND)
  saiga_vision_sample(sample_vision_arap.cpp)
endif()


if (SDL2_FOUND AND SAIGA_USE_OPENGL)
  saiga_vision_sample(sample_vision_DatasetViewer.cpp saiga_opengl)
  if(G2O_FOUND)
    saiga_vision_sample(sample_vision_ba.cpp saiga_opengl)
    saiga_vision_sample(sample_vision_posegraph_viewer.cpp saiga_opengl)
  endif()


  if(SAIGA_USE_K4A)
    saiga_vision_sample(sample_vision_KinectAzureViewer.cpp saiga_opengl)
  endif()
endif()

