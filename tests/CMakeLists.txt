find_package(GTest QUIET)
PackageHelperTarget(GTest::GTest GTest_FOUND)

if(NOT GTEST_FOUND)
  message(STATUS "GTest not found. Tests disabled.")
  return()
else()
endif()

function(saiga_test FILE_NAME)
  string(REGEX REPLACE "\\.[^.]*$" "" NAME ${FILE_NAME})

  set(TEST_TARGET ${NAME})

  add_executable(${TEST_TARGET} ${FILE_NAME})
  message(STATUS "Test enabled:      ${NAME}")


  target_link_libraries(${TEST_TARGET} PUBLIC saiga_core GTest::GTest GTest::Main)
  target_link_libraries(${TEST_TARGET} PUBLIC ${ARGN})

  add_test(
    NAME ${TEST_TARGET}
    COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TEST_TARGET}
    )


  set_property(TARGET ${TEST_TARGET} PROPERTY CUDA_ARCHITECTURES ${SAIGA_CUDA_ARCH})
  target_compile_options(${TEST_TARGET} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${SAIGA_CUDA_FLAGS}>)

  #set working directory for visual studio so the project can be executed from the ide
  set_target_properties(${TEST_TARGET} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${OUTPUT_DIR}")
  set_target_properties(${TEST_TARGET} PROPERTIES FOLDER tests/${PREFIX})
  set_target_properties(${TEST_TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
endfunction()


if(MODULE_CORE)
  saiga_test(test_core_align.cpp)
  saiga_test(test_core_frustum.cpp)
  saiga_test(test_core_vectorization.cpp)
  saiga_test(test_core_progressbar.cpp)

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
  saiga_test(test_vision_recursive_linear_systems.cpp "saiga_vision")
  if(K4A_FOUND)
    saiga_test(test_vision_azure.cpp "saiga_vision")
  endif()
endif()

if(MODULE_CUDA)
  saiga_test(test_cuda_image.cu "saiga_cuda")
  saiga_test(test_cuda_simple.cu "saiga_cuda")
  saiga_test(test_cuda_thrust.cu "saiga_cuda")
  if(MODULE_VISION)
    saiga_test(test_cuda_orb.cu "saiga_cuda" "saiga_vision")
  endif()
endif()








