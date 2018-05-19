# Locates the SAIGA library
# This module defines:
# SAIGA_FOUND
# SAIGA_INCLUDE_DIRS
# SAIGA_LIBRARY
#
# If SAIGA_NOT_ADD_DEPS is defined
# saigas additionally dependencies are not added 
# to SAIGA_INCLUDE_DIRS and SAIGA_LIBRARY

find_path(SAIGA_INCLUDE_DIRS 
	NAMES 
		saiga/rendering/deferred_renderer.h
	PATHS
          /usr/local/include
          /usr/include
	PATH_SUFFIXES
		  saiga/include
)

find_library(SAIGA_LIBRARY 
	NAMES 
		saiga
    PATHS 
		/usr/local/lib 
		/usr/lib
	PATH_SUFFIXES
		saiga/lib
)

if(SAIGA_INCLUDE_DIRS AND SAIGA_LIBRARY)
    set(SAIGA_FOUND TRUE)
endif()

if (SAIGA_FOUND)
    if(NOT SAIGA_FIND_QUIETLY)
        message(STATUS "Found saiga: ${SAIGA_INCLUDE_DIRS}")
    endif()
else()
    if(SAIGA_FIND_REQUIRED)
        message(FATAL_ERROR "Could not find saiga ${SAIGA_INCLUDE_DIRS} ${SAIGA_LIBRARY}")
    endif()
endif()


if(NOT SAIGA_NOT_ADD_DEPS)

#OPENGL
find_package(OpenGL REQUIRED QUIET)
SET(SAIGA_INCLUDE_DIRS ${SAIGA_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIRS})
SET(SAIGA_LIBRARY ${SAIGA_LIBRARY} ${OPENGL_LIBRARIES})

#GLM
find_package(GLM REQUIRED QUIET)
SET(SAIGA_INCLUDE_DIRS ${SAIGA_INCLUDE_DIRS} ${GLM_INCLUDE_DIRS})
include_directories( ${GLM_INCLUDE_DIRS}) 

#glbinding
find_package(glbinding QUIET)
if(GLBINDING_FOUND)
	SET(SAIGA_INCLUDE_DIRS ${SAIGA_INCLUDE_DIRS} ${GLBINDING_INCLUDE_DIRS})
	SET(SAIGA_LIBRARY ${SAIGA_LIBRARY} ${GLBINDING_LIBRARIES})
else()
#use GLEW as a fallback
#GLEW
find_package(GLEW QUIET)
if(GLEW_FOUND)
	SET(SAIGA_INCLUDE_DIRS ${SAIGA_INCLUDE_DIRS} ${GLEW_INCLUDE_DIRS})
	SET(SAIGA_LIBRARY ${SAIGA_LIBRARY} ${GLEW_LIBRARIES})
endif()
endif()


if(NOT GLBINDING_FOUND AND NOT GLEW_FOUND)
	message(FATAL_ERROR "No OpenGL loading library found.")
endif()

endif()



