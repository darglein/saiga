#
# Find FreeImage
#
# Try to find FreeImage.
# This module defines the following variables:
# - FREEIMAGE_INCLUDE_DIRS
# - FREEIMAGE_LIBRARIES
# - FREEIMAGE_FOUND
#
# The following variables can be set as arguments for the module.
# - FREEIMAGE_ROOT_DIR : Root library directory of FreeImage
#

# Additional modules
include(FindPackageHandleStandardArgs)

if (WIN32)
	# Find include files
	find_path(
		FREEIMAGE_INCLUDE_DIR
		NAMES FreeImage.h
		PATHS
			$ENV{PROGRAMFILES}/include
			${FREEIMAGE_ROOT_DIR}/include
		DOC "The directory where FreeImage.h resides")

	# Find library files
	find_library(
		FREEIMAGE_LIBRARY
		NAMES FreeImage
		PATHS
			$ENV{PROGRAMFILES}/lib
			${FREEIMAGE_ROOT_DIR}/lib)
else()
	# Find include files
	find_path(
		FREEIMAGE_INCLUDE_DIR
		NAMES FreeImage.h
		PATHS
			/usr/include
			/usr/local/include
			/sw/include
			/opt/local/include
		DOC "The directory where FreeImage.h resides")

	# Find library files
	find_library(
		FREEIMAGE_LIBRARY
		NAMES freeimage
		PATHS
			/usr/lib64
			/usr/lib
			/usr/local/lib64
			/usr/local/lib
			/sw/lib
			/opt/local/lib
			${FREEIMAGE_ROOT_DIR}/lib
		DOC "The FreeImage library")
endif()

if(FREEIMAGE_INCLUDE_DIR)
  foreach(freeimage_header FreeImage.h)
      file(STRINGS "${FREEIMAGE_INCLUDE_DIR}/${freeimage_header}" _freeimage_version_major   REGEX "^#define[\t ]+FREEIMAGE_MAJOR_VERSION[\t ]+.*")
      file(STRINGS "${FREEIMAGE_INCLUDE_DIR}/${freeimage_header}" _freeimage_version_minor   REGEX "^#define[\t ]+FREEIMAGE_MINOR_VERSION[\t ]+.*")
	  file(STRINGS "${FREEIMAGE_INCLUDE_DIR}/${freeimage_header}" _freeimage_version_release REGEX "^#define[\t ]+FREEIMAGE_RELEASE_SERIAL[\t ]+.*")

      string(REGEX REPLACE "^#define[\t ]+FREEIMAGE_MAJOR_VERSION[\t ]+(.+)" "\\1" _FREEIMAGE_MAJOR "${_freeimage_version_major}")
      string(REGEX REPLACE "^#define[\t ]+FREEIMAGE_MINOR_VERSION[\t ]+(.+)" "\\1" _FREEIMAGE_MINOR "${_freeimage_version_minor}")
      string(REGEX REPLACE "^#define[\t ]+FREEIMAGE_RELEASE_SERIAL[\t ]+(.+)" "\\1" _FREEIMAGE_RELEASE "${_freeimage_version_release}")
      unset(_freeimage_version_major)
      unset(_freeimage_version_minor)
      unset(_freeimage_version_release)

      set(FREEIMAGE_VERSION_STRING "${_FREEIMAGE_MAJOR}.${_FREEIMAGE_MINOR}.${_FREEIMAGE_RELEASE}")
      unset(_FREEIMAGE_MAJOR)
      unset(_FREEIMAGE_MINOR)
      unset(_FREEIMAGE_RELEASE)
      break()
  endforeach()
endif()

# Handle REQUIRD argument, define *_FOUND variable
find_package_handle_standard_args(FreeImage
									REQUIRED_VARS FREEIMAGE_INCLUDE_DIR FREEIMAGE_LIBRARY
									VERSION_VAR FREEIMAGE_VERSION_STRING)

if (FREEIMAGE_FOUND)
	set(FREEIMAGE_LIBRARIES ${FREEIMAGE_LIBRARY})
	set(FREEIMAGE_INCLUDE_DIRS ${FREEIMAGE_INCLUDE_DIR})
endif()

# Hide some variables
mark_as_advanced(FREEIMAGE_INCLUDE_DIR FREEIMAGE_LIBRARY)

