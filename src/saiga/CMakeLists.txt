# This file collects all source and header files
# The following variables are defined with "PARENT_SCOPE"
#
# SAIGA_ALL_FILES


add_subdirectory(animation)
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
    add_subdirectory(assets)
    add_subdirectory(opengl)
    add_subdirectory(rendering)
    add_subdirectory(smaa)
    add_subdirectory(world)
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

if (FFMPEG_FOUND)
    add_subdirectory(ffmpeg)
endif()

if (ASSIMP_FOUND)
    add_subdirectory(assimp)
endif()

if (EGL_FOUND)
    add_subdirectory(egl)
endif()

if(EIGEN3_FOUND)
    add_subdirectory(eigen)
endif()

if(FREETYPE_FOUND AND SAIGA_USE_OPENGL)
    add_subdirectory(text)
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
