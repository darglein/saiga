function(cuda_sample _name)

  set(TARGET_NAME "cuda_${_name}")
  message(STATUS "Sample enabled:      ${TARGET_NAME}")

  add_executable(${TARGET_NAME} "${_name}.cu" )

  # We only need to link the saiga target
  target_link_libraries(${TARGET_NAME} "saiga_cuda")
  target_include_directories(${TARGET_NAME} PUBLIC ".")

  #set_property(TARGET ${TARGET_NAME} PROPERTY CUDA_ARCHITECTURES 52)

  set_property(TARGET ${TARGET_NAME} PROPERTY CUDA_ARCHITECTURES ${SAIGA_CUDA_ARCH})
  target_compile_options(${TARGET_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${SAIGA_CUDA_FLAGS}>)


  #set working directory for visual studio so the project can be executed from the ide
  set_target_properties(${TARGET_NAME} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${OUTPUT_DIR}")
  set_target_properties(${TARGET_NAME} PROPERTIES FOLDER samples/cuda)
  set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
endfunction()



cuda_sample(hello_world)
cuda_sample(simple_particle)
cuda_sample(global_memory)
cuda_sample(random_access)
cuda_sample(constant_memory)
cuda_sample(async)
cuda_sample(event_sync)
cuda_sample(intra_block)
cuda_sample(bitonic_sort)
cuda_sample(bandwidth)
cuda_sample(radix_sort)
cuda_sample(reduce)

if(SAIGA_CUDA_RDC)
  cuda_sample(dynamic_parallelism)
  cuda_sample(iterative_sync)
endif()
