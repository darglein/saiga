
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
    set(OPUS_FOUND TRUE)
endif()

if (OPUS_FOUND)
    if(NOT Opus_FIND_QUIETLY)
        message(STATUS "Found glm: ${GLM_INCLUDE_DIRS}")
    endif()
else()
    if(Opus_FIND_REQUIRED)
        message(FATAL_ERROR "Could not find glm ${GLM_INCLUDE_DIRS}")
    endif()
endif()
