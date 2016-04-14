
############# Required Source Files ###############

FILE(GLOB_RECURSE animation_SRC  animation/*.cpp)
FILE(GLOB_RECURSE assets_SRC  assets/*.cpp)
FILE(GLOB_RECURSE camera_SRC  camera/*.cpp)
FILE(GLOB_RECURSE geometry_SRC  geometry/*.cpp)
FILE(GLOB_RECURSE opengl_SRC  opengl/*.cpp)
FILE(GLOB_RECURSE rendering_SRC  rendering/*.cpp)
FILE(GLOB_RECURSE text_SRC  text/*.cpp)
FILE(GLOB_RECURSE util_SRC  util/*.cpp)
FILE(GLOB_RECURSE window_SRC  window/*.cpp)
FILE(GLOB_RECURSE world_SRC  world/*.cpp)
FILE(GLOB main_SRC  *.cpp)

FILE(GLOB_RECURSE PROG_HEADER  ../${LIB_NAME}/*.h)

SET(PROG_SRC ${PROG_SRC} ${animation_SRC} ${assets_SRC} ${camera_SRC} ${geometry_SRC} ${opengl_SRC} 
${rendering_SRC} ${text_SRC} ${util_SRC} ${window_SRC} ${animation_SRC} ${world_SRC} ${main_SRC} ${PROG_HEADER})

############# Optional Source Files ###############

if (USE_SDL)
	FILE(GLOB SDL_SRC sdl/*.cpp)
	SET(PROG_SRC ${PROG_SRC} ${SDL_SRC})
endif (USE_SDL)

if (USE_GLFW)
	FILE(GLOB GLFW_SRC glfw/*.cpp)
	SET(PROG_SRC ${PROG_SRC} ${GLFW_SRC})
endif (USE_GLFW)

if (USE_OPENAL)
	FILE(GLOB OPENAL_SRC sound/*.cpp)
	SET(PROG_SRC ${PROG_SRC} ${OPENAL_SRC})
endif (USE_OPENAL)


#creates 'folders' in the solution explorer of visual studio
include(source_group_helper)
GroupSources2(${PROJECT_SOURCE_DIR}/src)
GroupSources2(${PROJECT_SOURCE_DIR}/saiga)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if(BUILD_SHARED)
	add_definitions(-DSAIGA_DLL_EXPORTS)
	add_library(${LIB_NAME} SHARED ${PROG_SRC})
else (BUILD_SHARED)
	add_library(${LIB_NAME} STATIC ${PROG_SRC})
endif(BUILD_SHARED)


target_link_libraries(${LIB_NAME} ${LIBS})




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


