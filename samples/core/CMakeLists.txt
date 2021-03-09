macro(saiga_core_sample FILE_NAME)
  string(REGEX REPLACE "\\.[^.]*$" "" NAME ${FILE_NAME})

  set(TARGET_NAME ${NAME})

  add_executable(${TARGET_NAME} ${FILE_NAME})
  message(STATUS "Sample enabled:      ${NAME}")


  target_link_libraries(${TARGET_NAME} PUBLIC saiga_core)
  target_link_libraries(${TARGET_NAME} PUBLIC ${ARGN})
  set_target_properties(${TARGET_NAME} PROPERTIES FOLDER samples/${PREFIX})
endmacro()


saiga_core_sample(sample_core_benchmark_disk.cpp)
saiga_core_sample(sample_core_benchmark_ipscaling.cpp)
saiga_core_sample(sample_core_benchmark_memcpy.cpp)
saiga_core_sample(sample_core_eigen.cpp)
saiga_core_sample(sample_core_filesystem.cpp)
saiga_core_sample(sample_core_fractals.cpp)
saiga_core_sample(sample_core_helloWorld.cpp)
saiga_core_sample(sample_core_imageprocessing.cpp)
saiga_core_sample(sample_core_ini.cpp)
saiga_core_sample(sample_core_nullspace.cpp)
saiga_core_sample(sample_core_openmp.cpp)
saiga_core_sample(sample_core_random.cpp)
saiga_core_sample(sample_core_raytracing.cpp)
saiga_core_sample(sample_core_stress.cpp)
saiga_core_sample(sample_core_threadpool.cpp)


