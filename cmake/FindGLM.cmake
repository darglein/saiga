
find_path(GLM_INCLUDE_DIRS 
	NAMES 
		glm.hpp
	PATHS
          /usr/local/include/opus
          /usr/local/include
          /usr/include/opus
          /usr/include
	PATH_SUFFIXES
		glm
)

if(GLM_INCLUDE_DIRS)
    set(GLM_FOUND TRUE)
else()
    set(GLM_FOUND FALSE)
endif()

if (GLM_FOUND)
    if(NOT GLM_FIND_QUIETLY)
        message(STATUS "Found glm: ${GLM_INCLUDE_DIRS}")
    endif()
else()
    if(GLM_FIND_REQUIRED)
        message(FATAL_ERROR "Could not find glm ${GLM_INCLUDE_DIRS}")
    endif()
endif()
mark_as_advanced(GLM_INCLUDE_DIRS)
