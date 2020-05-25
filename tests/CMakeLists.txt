find_package(GTest QUIET)

if(NOT GTEST_FOUND)
  message(STATUS "GTest not found. Tests disabled.")
  return()
endif()

function(saiga_test NAME)
  set(TEST_TARGET ${NAME})

  FILE(GLOB TEST_SRC  "${NAME}*")
  add_executable(${TEST_TARGET} ${TEST_SRC})
  message(STATUS "Test enabled:      ${NAME}")


  target_link_libraries(${TEST_TARGET} PUBLIC saiga_core GTest::GTest GTest::Main)
  target_link_libraries(${TEST_TARGET} PUBLIC ${ARGN})

  add_test(
    NAME ${TEST_TARGET}
    COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TEST_TARGET}
    )

  #set working directory for visual studio so the project can be executed from the ide
  set_target_properties(${PROG_NAME} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${OUTPUT_DIR}")
  set_target_properties(${PROG_NAME} PROPERTIES FOLDER tests/${PREFIX})
  set_target_properties(${PROG_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
endfunction()



if(MODULE_CORE)
  saiga_test(test_core_align)
  saiga_test(test_core_frustum)
  saiga_test(test_core_vectorization)

  if(OpenCV_FOUND AND MODULE_EXTRA)
    saiga_test(test_core_image_load_store ${EXTRA_LIBS})
  else()
    saiga_test(test_core_image_load_store)
  endif()
endif()
if(MODULE_VISION)
  if(CERES_FOUND)
    saiga_test(test_vision_bundle_adjustment "saiga_vision")
    saiga_test(test_vision_pose_graph_optimization "saiga_vision")
  endif()
  saiga_test(test_vision_bow "saiga_vision")
  saiga_test(test_vision_motion_model "saiga_vision")
  saiga_test(test_vision_pose_estimation "saiga_vision")
  saiga_test(test_vision_numeric_derivative "saiga_vision")
  saiga_test(test_vision_sophus "saiga_vision")
  saiga_test(test_vision_two_view_reconstruction "saiga_vision")
  saiga_test(test_vision_feature_grid "saiga_vision")
endif()

if(MODULE_CUDA)
  saiga_test(test_cuda_image "saiga_cuda")
  saiga_test(test_cuda_simple "saiga_cuda")
  saiga_test(test_cuda_thrust "saiga_cuda")
endif()








