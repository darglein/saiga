
macro(cuda_sample FILE_NAME)
  string(REGEX REPLACE "\\.[^.]*$" "" NAME ${FILE_NAME})

  set(TARGET_NAME ${NAME})

  # cuda architecture
  cmake_policy(SET CMP0104 OLD)
  add_executable(${TARGET_NAME} ${FILE_NAME})
  message(STATUS "Sample enabled:      ${NAME}")


  target_link_libraries(${TARGET_NAME} PUBLIC saiga_core saiga_cuda)
  target_link_libraries(${TARGET_NAME} PUBLIC ${ARGN})


  #set_property(TARGET ${TARGET_NAME} PROPERTY CUDA_ARCHITECTURES ${SAIGA_CUDA_ARCH})
  target_compile_options(${TARGET_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${SAIGA_CUDA_FLAGS}>)

  set_target_properties(${TARGET_NAME} PROPERTIES FOLDER samples/${PREFIX})
endmacro()




cuda_sample(sample_cuda_hello_world.cu)
cuda_sample(sample_cuda_simple_particle.cu)
cuda_sample(sample_cuda_global_memory.cu)
cuda_sample(sample_cuda_random_access.cu)
cuda_sample(sample_cuda_constant_memory.cu)
cuda_sample(sample_cuda_async.cu)
cuda_sample(sample_cuda_event_sync.cu)
cuda_sample(sample_cuda_intra_block.cu)
cuda_sample(sample_cuda_bitonic_sort.cu)
cuda_sample(sample_cuda_bandwidth.cu)
cuda_sample(sample_cuda_radix_sort.cu)
cuda_sample(sample_cuda_reduce.cu)

if(SAIGA_CUDA_RDC)
  cuda_sample(sample_cuda_dynamic_parallelism.cu)
  cuda_sample(sample_cuda_iterative_sync.cu)
endif()
