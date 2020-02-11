add_subdirectory(benchmarkIPScaling)
add_subdirectory(benchmarkMemcpy)
add_subdirectory(benchmarkDisk)
add_subdirectory(nullspace)

add_subdirectory(helloWorld)
add_subdirectory(raytracing)
add_subdirectory(ini)
add_subdirectory(threadPool)
add_subdirectory(randomNumbers)
add_subdirectory(imageProcessing)
add_subdirectory(fractals)
add_subdirectory(filesystem)

add_subdirectory(eigen)
add_subdirectory(stress_test)

if(OPENMP_FOUND)
    add_subdirectory(openmp)
endif()



