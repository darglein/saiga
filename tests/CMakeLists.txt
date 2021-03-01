#find_package(GTest REQUIRED)
find_package(GTest CONFIG REQUIRED)
PackageHelperTarget(GTest::gtest GTEST_FOUND)

if(NOT GTEST_FOUND)
  message(STATUS "GTest not found. Tests disabled.")
  return()
else()
endif()

function(saiga_test FILE_NAME)
  string(REGEX REPLACE "\\.[^.]*$" "" NAME ${FILE_NAME})

  set(TARGET_NAME ${NAME})

  add_executable(${TARGET_NAME} ${FILE_NAME})
  message(STATUS "Test enabled:      ${NAME}")


  target_link_libraries(${TARGET_NAME} PUBLIC saiga_core GTest::gtest GTest::gtest_main)
  target_link_libraries(${TARGET_NAME} PUBLIC ${ARGN})

  add_test(
    NAME ${TARGET_NAME}
    COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET_NAME}
    )


  set_property(TARGET ${TARGET_NAME} PROPERTY CUDA_ARCHITECTURES ${SAIGA_CUDA_ARCH})
  target_compile_options(${TARGET_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${SAIGA_CUDA_FLAGS}>)

  #set working directory for visual studio so the project can be executed from the ide
  set_target_properties(${TARGET_NAME} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${OUTPUT_DIR}")
  set_target_properties(${TARGET_NAME} PROPERTIES FOLDER tests/${PREFIX})
  set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
endfunction()


if(MODULE_CORE)
  saiga_test(test_core_align.cpp)
  saiga_test(test_core_frustum.cpp)
  if(SAIGA_USE_ZLIB)
    saiga_test(test_core_zlib.cpp)
  endif()
  saiga_test(test_core_vectorization.cpp)
  saiga_test(test_core_progressbar.cpp)
  saiga_test(test_core_rectangular_decomposition.cpp)

  if(OpenCV_FOUND AND MODULE_EXTRA)
    saiga_test(test_core_image_load_store.cpp ${EXTRA_LIBS})
  else()
    saiga_test(test_core_image_load_store.cpp)
  endif()
endif()
if(MODULE_VISION)
  if(CERES_FOUND)
    saiga_test(test_vision_bundle_adjustment.cpp "saiga_vision")
    saiga_test(test_vision_bundle_adjustment_relpose.cpp "saiga_vision")
    saiga_test(test_vision_pose_graph_optimization.cpp "saiga_vision")
    saiga_test(test_vision_pose_alignment.cpp "saiga_vision")
    saiga_test(test_vision_imu_decoupled_solver.cpp "saiga_vision")
    saiga_test(test_vision_pose_estimation.cpp "saiga_vision")
  endif()
  saiga_test(test_vision_bow.cpp "saiga_vision")
  saiga_test(test_vision_nearest_neighbor.cpp "saiga_vision")
  saiga_test(test_vision_poisson_surface.cpp "saiga_vision")
  saiga_test(test_vision_distortion.cpp "saiga_vision")
  saiga_test(test_vision_motion_model.cpp "saiga_vision")
  saiga_test(test_vision_numeric_derivative.cpp "saiga_vision")
  saiga_test(test_vision_derivative_relpose.cpp "saiga_vision")
  saiga_test(test_vision_sophus.cpp "saiga_vision")
  saiga_test(test_vision_two_view_reconstruction.cpp "saiga_vision")
  saiga_test(test_vision_feature_grid.cpp "saiga_vision")
  saiga_test(test_vision_five_eight_point.cpp "saiga_vision")
  saiga_test(test_vision_imu.cpp "saiga_vision")
  saiga_test(test_vision_imu_derivatives.cpp "saiga_vision")
  saiga_test(test_vision_robust_cost_function.cpp "saiga_vision")
  saiga_test(test_vision_tsdf.cpp "saiga_vision")
  saiga_test(test_vision_tsdf_fuse.cpp "saiga_vision")
  saiga_test(test_vision_recursive_linear_systems.cpp "saiga_vision")
  if(K4A_FOUND)
    saiga_test(test_vision_azure.cpp "saiga_vision")
  endif()
endif()

if(MODULE_CUDA)
  saiga_test(test_cuda_image.cu "saiga_cuda")
  saiga_test(test_cuda_simple.cu "saiga_cuda")
  saiga_test(test_cuda_thrust.cu "saiga_cuda")
  saiga_test(test_cuda_bvh.cu "saiga_cuda")
  if(MODULE_VISION)
    saiga_test(test_cuda_orb.cu "saiga_cuda" "saiga_vision")
  endif()
endif()








