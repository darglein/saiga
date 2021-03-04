macro(saiga_vulkan_sample FILE_NAME)
  string(REGEX REPLACE "\\.[^.]*$" "" NAME ${FILE_NAME})

  set(TARGET_NAME ${NAME})

  add_executable(${TARGET_NAME} ${FILE_NAME})
  message(STATUS "Sample enabled:      ${NAME}")


  target_link_libraries(${TARGET_NAME} PUBLIC saiga_core saiga_vulkan)
  target_link_libraries(${TARGET_NAME} PUBLIC ${ARGN})

  #set working directory for visual studio so the project can be executed from the ide
  set_target_properties(${TARGET_NAME} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${OUTPUT_DIR}")
  set_target_properties(${TARGET_NAME} PROPERTIES FOLDER tests/${PREFIX})
  set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
endmacro()


saiga_vulkan_sample(sample_vulkan_compute.cpp)
saiga_vulkan_sample(sample_vulkan_helloworld.cpp)
saiga_vulkan_sample(sample_vulkan_memory.cpp)
