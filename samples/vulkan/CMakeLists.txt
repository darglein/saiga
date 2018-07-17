add_subdirectory(helloVulkan)
add_subdirectory(compute)
add_subdirectory(swapTest)

if(OPENNI2_FOUND)
    add_subdirectory(openni2Viewer)
endif()

