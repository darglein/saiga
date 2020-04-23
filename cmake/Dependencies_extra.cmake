# Defines the following output variables:
#
# EXTRA_INCLUDES:    The list of required include directories
# EXTRA_LIBS:        The list of required libraries for link_target
# EXTRA_TARGETS:     The list of required targets
# MODULE_EXTRA:      True if all required dependencies are found.
#

unset(PACKAGE_INCLUDES)
unset(LIB_TARGETS)
unset(LIBS)
unset(MODULE_EXTRA)


if(NOT MODULE_CORE)
    return()
endif()



#opencv
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    SET(SAIGA_USE_OPENCV 1)
endif()
PackageHelper(OpenCV "${OpenCV_FOUND}" "${OpenCV_INCLUDE_DIRS}" "${OpenCV_LIBRARIES}")


#gphoto2
find_package(GPHOTO2 QUIET)
PackageHelper(GPHOTO2 "${GPHOTO2_FOUND}" "${Gphoto2_INCLUDE_DIRS}" "${Gphoto2_LIBRARIES}")
if(GPHOTO2_FOUND)
    SET(SAIGA_USE_GPHOTO2 1)
endif()



set(EXTRA_INCLUDES ${PACKAGE_INCLUDES})
set(EXTRA_LIBS ${LIBS})
set(EXTRA_TARGETS saiga_core ${LIB_TARGETS})
set(MODULE_EXTRA 1)

