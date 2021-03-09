macro(saiga_vulkan_sample FILE_NAME)
  string(REGEX REPLACE "\\.[^.]*$" "" NAME ${FILE_NAME})

  set(TARGET_NAME ${NAME})

  add_executable(${TARGET_NAME} ${FILE_NAME})
  message(STATUS "Sample enabled:      ${NAME}")


  target_link_libraries(${TARGET_NAME} PUBLIC saiga_core saiga_vulkan)
  target_link_libraries(${TARGET_NAME} PUBLIC ${ARGN})

  set_target_properties(${TARGET_NAME} PROPERTIES FOLDER samples/${PREFIX})
endmacro()


saiga_vulkan_sample(sample_vulkan_compute.cpp)
saiga_vulkan_sample(sample_vulkan_helloworld.cpp)
saiga_vulkan_sample(sample_vulkan_memory.cpp)
