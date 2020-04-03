find_package(GTest QUIET)

if(NOT GTEST_FOUND)
  message(STATUS "GTest not found. Tests disabled.")
  return()
endif()

function(saiga_test NAME)
  set(TEST_TARGET test_${NAME})
  add_executable(${TEST_TARGET} ${NAME}_test.cpp)
  message(STATUS "Test enabled:      ${NAME}")


  target_link_libraries(${TEST_TARGET} PUBLIC saiga_core GTest::GTest GTest::Main)
  target_link_libraries(${TEST_TARGET} PUBLIC ${ARGN})

  add_test(
    NAME ${TEST_TARGET}
    COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${PROG_NAME}
    )

  #set working directory for visual studio so the project can be executed from the ide
  set_target_properties(${PROG_NAME} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${OUTPUT_DIR}")
  set_target_properties(${PROG_NAME} PROPERTIES FOLDER tests/${PREFIX})
  set_target_properties(${PROG_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
endfunction()



if(MODULE_CORE)
  saiga_test(core_align)
  saiga_test(core_frustum)
  saiga_test(core_imageloadstore)
endif()
if(MODULE_VISION)
  if(CERES_FOUND)
    saiga_test(vision_BundleAdjustment "saiga_vision")
    saiga_test(vision_PoseGraphOptimization "saiga_vision")
  endif()
  saiga_test(vision_motionModel "saiga_vision")
  saiga_test(vision_PoseEstimation "saiga_vision")
  saiga_test(vision_numeric_derivative "saiga_vision")
endif()








