
############# Required Source Files ###############

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
FILE(GLOB_RECURSE text_SRC  text/*.cpp)
FILE(GLOB_RECURSE tests_SRC  tests/*.cpp)
FILE(GLOB_RECURSE util_SRC  util/*.cpp)
FILE(GLOB_RECURSE window_SRC  window/*.cpp)
FILE(GLOB_RECURSE world_SRC  world/*.cpp)
FILE(GLOB main_SRC  *.cpp)

FILE(GLOB_RECURSE PROG_HEADER  ${PROJECT_SOURCE_DIR}/include/${LIB_NAME}/*.h)

FILE(GLOB_RECURSE SHADERS  ${PROJECT_SOURCE_DIR}/shader/*.glsl)

SET(PROG_SRC ${PROG_SRC} ${animation_SRC} ${assets_SRC} ${camera_SRC} ${eigen_SRC} ${geometry_SRC} ${opengl_SRC} ${image_SRC} ${imgui_SRC} ${rendering_SRC} ${smaa_SRC} ${time_SRC} ${text_SRC} ${tests_SRC} ${util_SRC} ${window_SRC} ${animation_SRC} ${world_SRC} ${main_SRC} ${PROG_HEADER})

#add shaders to sources so they are added to the msvc solution.
SET(PROG_SRC ${PROG_SRC} ${SHADERS})

############# Optional Source Files ###############

if (SDL2_FOUND)
	FILE(GLOB SDL_SRC sdl/*.cpp)
	SET(PROG_SRC ${PROG_SRC} ${SDL_SRC})
endif ()

if (GLFW_FOUND)
	FILE(GLOB GLFW_SRC glfw/*.cpp)
	SET(PROG_SRC ${PROG_SRC} ${GLFW_SRC})
endif ()

if (OPENAL_FOUND)
	FILE(GLOB OPENAL_SRC sound/*.cpp)
	SET(PROG_SRC ${PROG_SRC} ${OPENAL_SRC})
endif ()

if (FFMPEG_FOUND)
	FILE(GLOB_RECURSE ffmpeg_SRC  ffmpeg/*.cpp)
	SET(PROG_SRC ${PROG_SRC} ${ffmpeg_SRC})
endif()

if (ASSIMP_FOUND)
	FILE(GLOB_RECURSE assimp_SRC  assimp/*.cpp)
	SET(PROG_SRC ${PROG_SRC} ${assimp_SRC})
endif()

if (EGL_FOUND)
	FILE(GLOB_RECURSE egl_SRC  egl/*.cpp)
	SET(PROG_SRC ${PROG_SRC} ${egl_SRC})
endif()

if(EIGEN_FOUND)
	FILE(GLOB_RECURSE eigen_SRC  eigen/*.cpp)
	SET(PROG_SRC ${PROG_SRC} ${eigen_SRC})
endif()

if(CUDA_FOUND)
	FILE(GLOB_RECURSE CUDA_SRC  cuda/*.cu)
	cuda_compile(cuda_objs ${CUDA_SRC})
	SET(PROG_SRC ${PROG_SRC} ${cuda_objs})
endif()





#creates 'folders' in the solution explorer of visual studio
include(source_group_helper)
GroupSources2(${PROJECT_SOURCE_DIR}/src)
GroupSources2(${PROJECT_SOURCE_DIR}/include/saiga)
GroupSources(${PROJECT_SOURCE_DIR} shader)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/include/saiga/imgui)

if(BUILD_SHARED)
	add_definitions(-DSAIGA_DLL_EXPORTS)
	add_library(${LIB_NAME} SHARED ${PROG_SRC})
else (BUILD_SHARED)
	add_library(${LIB_NAME} STATIC ${PROG_SRC})
endif(BUILD_SHARED)


target_link_libraries(${LIB_NAME} ${LIBS})

set_target_properties( ${LIB_NAME}
	PROPERTIES
    	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")


if(BUILD_SHARED)
    install(
	TARGETS ${LIB_NAME} 
        RUNTIME DESTINATION ${INSTALL_BIN}
        LIBRARY DESTINATION ${INSTALL_SHARED}
	)
else()
    install(
	TARGETS ${LIB_NAME} 
        ARCHIVE DESTINATION ${INSTALL_LIB}
	)
endif()




