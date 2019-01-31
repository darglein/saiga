# This file collects all source and header files
# The following variables are defined with "PARENT_SCOPE"
#
# SAIGA_ALL_FILES

add_subdirectory(core)
add_subdirectory(vulkan)
add_subdirectory(opengl)
add_subdirectory(extra)
add_subdirectory(cuda)
add_subdirectory(vision)

return()

#add_subdirectory(animation)
add_subdirectory(camera)
add_subdirectory(framework)
add_subdirectory(model)
add_subdirectory(window)
add_subdirectory(image)
add_subdirectory(time)
add_subdirectory(geometry)
add_subdirectory(util)
add_subdirectory(tests)
add_subdirectory(imgui)


if(SAIGA_USE_OPENGL)
    add_subdirectory(opengl)
endif()


if (SDL2_FOUND)
    add_subdirectory(sdl)
endif ()

if (GLFW_FOUND)
    add_subdirectory(glfw)
endif ()

if (OPENAL_FOUND)
    add_subdirectory(sound)
endif ()

if(EIGEN3_FOUND)
    add_subdirectory(eigen)
endif()



if(CUDA_FOUND)
    add_subdirectory(cuda)
endif()

if(SAIGA_USE_VULKAN)
    add_subdirectory(vulkan)
endif()

if(OPENNI2_FOUND)
    add_subdirectory(openni2)
endif()

if(Boost_FOUND)
    add_subdirectory(network)
endif()

if(OpenCV_FOUND)
    add_subdirectory(opencv)
endif()

if(GPHOTO2_FOUND)
    add_subdirectory(gphoto2)
endif()

if(SAIGA_VISION)
    add_subdirectory(vision)
endif()


FILE(GLOB BASE_SRC  "*.cpp" "*.h")
set(SAIGA_ALL_FILES ${SAIGA_ALL_FILES} ${BASE_SRC} PARENT_SCOPE)
