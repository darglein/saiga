add_subdirectory(helloVulkan)
add_subdirectory(vulkanMemoryTest)
add_subdirectory(compute)
add_subdirectory(swapTest)

if(OPENNI2_FOUND AND SAIGA_VISION)
    add_subdirectory(openni2Viewer)
endif()

