#
# Try to find the FreeImage library and include path.
# Once done this will define
#
# FREEIMAGEPLUS_FOUND
# FREEIMAGEPLUS_INCLUDE_PATH
# FREEIMAGEPLUS_LIBRARY
# FREEIMAGEPLUS_LIBRARIES
#

IF (WIN32)
	FIND_PATH( FREEIMAGEPLUS_INCLUDE_PATH FreeImagePlus.h
		${FREEIMAGE_ROOT_DIR}/include
		${FREEIMAGE_ROOT_DIR}
		DOC "The directory where FreeImagePlus.h resides")
	FIND_LIBRARY( FREEIMAGEPLUS_LIBRARY
		NAMES FreeImagePlus freeimageplus
		PATHS
		${FREEIMAGE_ROOT_DIR}/lib
		${FREEIMAGE_ROOT_DIR}
		DOC "The FreeImagePlus library")
ELSE (WIN32)
	FIND_PATH( FREEIMAGEPLUS_INCLUDE_PATH FreeImagePlus.h
		/usr/include
		/usr/local/include
		/sw/include
		/opt/local/include
		DOC "The directory where FreeImagePlus.h resides")
	FIND_LIBRARY( FREEIMAGEPLUS_LIBRARY
		NAMES FreeImagePlus freeimageplus
		PATHS
		/usr/lib64
		/usr/lib
		/usr/local/lib64
		/usr/local/lib
		/sw/lib
		/opt/local/lib
		DOC "The FreeImagePlus library")
ENDIF (WIN32)


if(FREEIMAGEPLUS_INCLUDE_PATH)
  foreach(freeimage_header FreeImage.h)
      file(STRINGS "${FREEIMAGEPLUS_INCLUDE_PATH}/${freeimage_header}" _freeimage_version_major   REGEX "^#define[\t ]+FREEIMAGE_MAJOR_VERSION[\t ]+.*")
      file(STRINGS "${FREEIMAGEPLUS_INCLUDE_PATH}/${freeimage_header}" _freeimage_version_minor   REGEX "^#define[\t ]+FREEIMAGE_MINOR_VERSION[\t ]+.*")
	  file(STRINGS "${FREEIMAGEPLUS_INCLUDE_PATH}/${freeimage_header}" _freeimage_version_release REGEX "^#define[\t ]+FREEIMAGE_RELEASE_SERIAL[\t ]+.*")

      string(REGEX REPLACE "^#define[\t ]+FREEIMAGE_MAJOR_VERSION[\t ]+(.+)" "\\1" _FREEIMAGE_MAJOR "${_freeimage_version_major}")
      string(REGEX REPLACE "^#define[\t ]+FREEIMAGE_MINOR_VERSION[\t ]+(.+)" "\\1" _FREEIMAGE_MINOR "${_freeimage_version_minor}")
      string(REGEX REPLACE "^#define[\t ]+FREEIMAGE_RELEASE_SERIAL[\t ]+(.+)" "\\1" _FREEIMAGE_RELEASE "${_freeimage_version_release}")
      unset(_freeimage_version_major)
      unset(_freeimage_version_minor)
      unset(_freeimage_version_release)

      set(FREEIMAGEPLUS_VERSION_STRING "${_FREEIMAGE_MAJOR}.${_FREEIMAGE_MINOR}.${_FREEIMAGE_RELEASE}")
      unset(_FREEIMAGE_MAJOR)
      unset(_FREEIMAGE_MINOR)
      unset(_FREEIMAGE_RELEASE)
      break()
  endforeach()
endif()


# Handle REQUIRD argument, define *_FOUND variable
find_package_handle_standard_args(FreeImagePlus
									REQUIRED_VARS FREEIMAGEPLUS_INCLUDE_PATH FREEIMAGEPLUS_LIBRARY
									VERSION_VAR FREEIMAGEPLUS_VERSION_STRING)

if (FREEIMAGEPLUS_FOUND)
	set(FREEIMAGEPLUS_LIBRARIES ${FREEIMAGEPLUS_LIBRARY})
	set(FREEIMAGEPLUS_INCLUDE_DIRS ${FREEIMAGEPLUS_INCLUDE_DIR})
endif()

# Hide some variables
mark_as_advanced(
	FREEIMAGEPLUS_FOUND
	FREEIMAGEPLUS_LIBRARY
	FREEIMAGEPLUS_LIBRARIES
	FREEIMAGEPLUS_INCLUDE_PATH)

# SET(FREEIMAGEPLUS_LIBRARIES ${FREEIMAGEPLUS_LIBRARY})

# IF (FREEIMAGEPLUS_INCLUDE_PATH AND FREEIMAGEPLUS_LIBRARY)
# 	SET( FREEIMAGEPLUS_FOUND TRUE)
# ELSE ()
# 	SET( FREEIMAGEPLUS_FOUND FALSE)
# ENDIF ()

# MARK_AS_ADVANCED(
# 	FREEIMAGEPLUS_FOUND
# 	FREEIMAGEPLUS_LIBRARY
# 	FREEIMAGEPLUS_LIBRARIES
# 	FREEIMAGEPLUS_INCLUDE_PATH)


