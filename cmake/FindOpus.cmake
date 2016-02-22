if(Opus_INCLUDE_DIR AND Opus_LIBRARY)
    set(Opus_FIND_QUIETLY TRUE)
endif()

find_path(OPUS_INCLUDE_DIRS opus.h
          /usr/local/include/opus
          /usr/local/include
          /usr/include/opus
          /usr/include
)

find_path(OPUS_FILE_INCLUDE_DIRS opusfile.h
          /usr/local/include/opus
          /usr/local/include
          /usr/include/opus
          /usr/include
)

find_library(OPUS_LIBRARY NAMES opus
             PATHS /usr/local/lib /usr/lib)

find_library(OPUS_FILE_LIBRARY NAMES opusfile
             PATHS /usr/local/lib /usr/lib)


set(OPUS_LIBRARIES
    ${OPUS_LIBRARY}
	${OPUS_FILE_LIBRARY}
)

if(OPUS_INCLUDE_DIRS AND OPUS_LIBRARY AND OPUS_FILE_INCLUDE_DIRS AND OPUS_FILE_LIBRARY)
    set(OPUS_FOUND TRUE)
endif()

if (OPUS_FOUND)
    if(NOT Opus_FIND_QUIETLY)
        message(STATUS "Found libopus: ${OPUS_INCLUDE_DIRS}")
    endif()
else()
    if(Opus_FIND_REQUIRED)
        message(FATAL_ERROR "Could not find libopus")
    endif()
endif()

