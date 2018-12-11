add_subdirectory(vulkan_compute)
add_subdirectory(vulkan_helloWorld)
add_subdirectory(vulkan_memoryTest)
add_subdirectory(vulkan_swapTest)

if(OPENNI2_FOUND AND SAIGA_VISION)
    add_subdirectory(vulkan_openni2Viewer)
endif()

