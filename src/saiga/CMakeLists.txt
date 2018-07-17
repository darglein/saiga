# This file collects all source and header files
# The following variables are defined with "PARENT_SCOPE"
#
# SAIGA_SOURCE_CORE	-	source files that are always compiled
# SAIGA_SOURCE_EXT	-	source files from optional dependencies
# SAIGA_HEADERS
# SAIGA_SHADERS

############# Core Source Files ###############

FILE(GLOB_RECURSE animation_SRC  animation/*.cpp)
FILE(GLOB_RECURSE assets_SRC  assets/*.cpp)
FILE(GLOB_RECURSE camera_SRC  camera/*.cpp)
FILE(GLOB_RECURSE geometry_SRC  geometry/*.cpp)
FILE(GLOB_RECURSE opengl_SRC  opengl/*.cpp)
FILE(GLOB_RECURSE image_SRC  image/*.cpp)
FILE(GLOB_RECURSE imgui_SRC  imgui/*.cpp)
FILE(GLOB_RECURSE rendering_SRC  rendering/*.cpp)
FILE(GLOB_RECURSE smaa_SRC  smaa/*.cpp)
FILE(GLOB_RECURSE time_SRC  time/*.cpp)
FILE(GLOB_RECURSE tests_SRC  tests/*.cpp)
FILE(GLOB_RECURSE util_SRC  util/*.cpp)
FILE(GLOB_RECURSE window_SRC  window/*.cpp)
FILE(GLOB_RECURSE world_SRC  world/*.cpp)
FILE(GLOB main_SRC  *.cpp)

SET(SAIGA_SOURCE_CORE 
    ${animation_SRC}
    ${assets_SRC}
    ${camera_SRC}
    ${eigen_SRC}
    ${geometry_SRC}
    ${opengl_SRC}
    ${image_SRC}
    ${imgui_SRC}
    ${rendering_SRC}
    ${smaa_SRC}
    ${time_SRC}
    ${tests_SRC}
    ${util_SRC}
    ${window_SRC}
    ${animation_SRC}
    ${world_SRC}
    ${main_SRC}
    ${PROG_HEADER}
    PARENT_SCOPE
    )

############# Optional Source Files ###############

if (SDL2_FOUND)
    FILE(GLOB SDL_SRC sdl/*.cpp)
endif ()

if (GLFW_FOUND)
    FILE(GLOB GLFW_SRC glfw/*.cpp)
endif ()

if (OPENAL_FOUND)
    FILE(GLOB OPENAL_SRC sound/*.cpp)
endif ()

if (FFMPEG_FOUND)
    FILE(GLOB_RECURSE ffmpeg_SRC  ffmpeg/*.cpp)
endif()

if (ASSIMP_FOUND)
    FILE(GLOB_RECURSE assimp_SRC  assimp/*.cpp)
endif()

if (EGL_FOUND)
    FILE(GLOB_RECURSE egl_SRC  egl/*.cpp)
endif()

if(EIGEN3_FOUND)
    FILE(GLOB_RECURSE eigen_SRC  eigen/*.cpp)
endif()

if(FREETYPE_FOUND)
    FILE(GLOB_RECURSE text_SRC  text/*.cpp)
endif()

if(CUDA_FOUND)
    FILE(GLOB_RECURSE CUDA_SRC  cuda/*.cu)
endif()

if(SAIGA_USE_VULKAN)
    FILE(GLOB_RECURSE vulkan_SRC  vulkan/*.cpp)
endif()

if(OPENNI2_FOUND)
    FILE(GLOB_RECURSE openni2_SRC  openni2/*.cpp)
endif()


SET(SAIGA_SOURCE_EXT
    ${SDL_SRC}
    ${GLFW_SRC}
    ${OPENAL_SRC}
    ${ffmpeg_SRC}
    ${assimp_SRC}
    ${egl_SRC}
    ${eigen_SRC}
    ${text_SRC}
    ${CUDA_SRC}
    ${vulkan_SRC}
    ${openni2_SRC}
    PARENT_SCOPE
    )

############# All Shader Files ###############

FILE(GLOB_RECURSE SHADERS  ${PROJECT_SOURCE_DIR}/shader/*.glsl)
set(SAIGA_SHADERS ${SHADERS} PARENT_SCOPE)

############# All Header Files ###############

FILE(GLOB_RECURSE PROG_HEADER  ${PROJECT_SOURCE_DIR}/src/${LIB_NAME}/*.h)
FILE(GLOB_RECURSE PROG_HEADER2  ${PROJECT_SOURCE_DIR}/src/${LIB_NAME}/*.hpp)
set(SAIGA_HEADERS ${PROG_HEADER} ${PROG_HEADER2} PARENT_SCOPE)

