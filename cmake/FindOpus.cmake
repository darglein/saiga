if(Opus_INCLUDE_DIR AND Opus_LIBRARY)
    set(Opus_FIND_QUIETLY TRUE)
endif()

find_path(OPUS_INCLUDE_DIRS 
	NAMES 
		opus.h
	PATHS
          /usr/local/include/opus
          /usr/local/include
          /usr/include/opus
          /usr/include
	PATH_SUFFIXES
		opus
)

if(NOT OPUS_INCLUDE_DIRS AND NOT Opus_FIND_QUIETLY)
	message(FATAL_ERROR "Could not find libopus ${OPUS_INCLUDE_DIRS}")
endif()

find_library(OPUS_LIBRARY 
	NAMES 
		opus
    PATHS 
		/usr/local/lib 
		/usr/lib
)

if(NOT OPUS_LIBRARY AND NOT Opus_FIND_QUIETLY)
	message(FATAL_ERROR "Could not find libopus ${OPUS_LIBRARY}")
endif()
             

find_path(OPUS_FILE_INCLUDE_DIRS 
	NAMES
		opusfile.h
	PATHS
          /usr/local/include/opus
          /usr/local/include
          /usr/include/opus
          /usr/include
	PATH_SUFFIXES
		opus
)

find_library(OPUS_FILE_LIBRARY NAMES opusfile
             PATHS /usr/local/lib /usr/lib)

if( (NOT OPUS_FILE_INCLUDE_DIRS OR NOT OPUS_FILE_LIBRARY)  AND NOT Opus_FIND_QUIETLY)
	message(FATAL_ERROR "Could not find opusfile ${OPUS_FILE_INCLUDE_DIRS}, ${OPUS_FILE_LIBRARY}")
endif()
             
#opusfile requires libopus
find_path(OGG_INCLUDE_DIRS 
	NAMES	
		ogg/ogg.h
	PATHS
          /usr/local/include/opus
          /usr/local/include
          /usr/include/opus
          /usr/include
)
find_library(OGG_LIBRARY 
	NAMES 
		ogg
		libogg_static
    PATHS 
		/usr/local/lib 
		/usr/lib
)

if( (NOT OGG_INCLUDE_DIRS OR NOT OGG_LIBRARY) AND NOT Opus_FIND_QUIETLY)
	message(FATAL_ERROR "Could not find ogg ${OGG_INCLUDE_DIRS}, ${OGG_LIBRARY}")
endif()
          
set(OPUS_LIBRARIES
    	${OPUS_LIBRARY}
	${OPUS_FILE_LIBRARY}
   	${OGG_LIBRARY}
)

if(WIN32)   
#celt codec
find_library(CELT_LIBRARY NAMES celt
             PATHS /usr/local/lib /usr/lib)     
#silk codec
find_library(SILK_LIBRARY NAMES silk_common
             PATHS /usr/local/lib /usr/lib)
set(OPUS_LIBRARIES
	${OPUS_LIBRARIES}
    	${CELT_LIBRARY}
    	${SILK_LIBRARY}
)
endif()
             
             

if(OPUS_INCLUDE_DIRS AND OPUS_LIBRARY AND OPUS_FILE_INCLUDE_DIRS AND OPUS_FILE_LIBRARY AND OGG_LIBRARY AND 
(NOT WIN32 OR (CELT_LIBRARY AND SILK_LIBRARY)))
    set(OPUS_FOUND TRUE)
else()
	set(OPUS_FOUND FALSE)
endif()

if (OPUS_FOUND)
    if(NOT Opus_FIND_QUIETLY)
        message(STATUS "Found libopus: ${OPUS_INCLUDE_DIRS}")
    endif()
else()
    if(Opus_FIND_REQUIRED)
        message(FATAL_ERROR "Could not find libopus ${OPUS_LIBRARIES}")
    endif()
endif()

mark_as_advanced(OPUS_INCLUDE_DIRS OPUS_LIBRARIES OPUS_LIBRARY OPUS_FILE_LIBRARY OGG_LIBRARY OGG_INCLUDE_DIRS OPUS_FILE_INCLUDE_DIRS)

