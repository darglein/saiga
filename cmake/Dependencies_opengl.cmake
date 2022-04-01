# Defines the following output variables:
#
# OPENGL_INCLUDES:    The list of required include directories
# OPENGL_LIBS:        The list of required libraries for link_target
# OPENGL_TARGETS:     The list of required targets
# MODULE_OPENGL:      True if all required dependencies are found.
#

unset(PACKAGE_INCLUDES)
unset(LIB_TARGETS)
unset(LIBS)
unset(MODULE_OPENGL)

if (NOT MODULE_CORE)
    return()
endif ()


if (SAIGA_MODULE_OPENGL)
    set(OpenGL_GL_PREFERENCE LEGACY)
    find_package(OpenGL)
    PackageHelper(OpenGL "${OPENGL_FOUND}" "${OPENGL_INCLUDE_DIR}" "${OPENGL_LIBRARIES}")
    #PackageHelperTarget(OpenGL::GL OPENGL_FOUND)
    if (OPENGL_FOUND)
        SET(SAIGA_USE_OPENGL 1)
    else ()
        return()
    endif ()
else ()
    UNSET(SAIGA_USE_OPENGL)
    return()
endif ()


#freetype2
if (SAIGA_WITH_FREETYPE)
    find_package(Freetype QUIET)
    PackageHelper(Freetype "${FREETYPE_FOUND}" "${FREETYPE_INCLUDE_DIRS}" "${FREETYPE_LIBRARIES}")
    if (FREETYPE_FOUND)
        SET(SAIGA_USE_FREETYPE 1)
    else ()
        SET(SAIGA_USE_FREETYPE 0)
    endif ()
endif ()

if (SAIGA_WITH_BULLET)
    #bullet
    find_package(Bullet QUIET)
    if (BULLET_FOUND)
        SET(SAIGA_USE_BULLET 1)
    endif ()
    PackageHelper(Bullet "${BULLET_FOUND}" "${BULLET_INCLUDE_DIR}" "${BULLET_LIBRARIES}")
endif ()


#EGL
find_package(EGL QUIET)
if (EGL_FOUND)
    SET(SAIGA_USE_EGL 1)
endif ()
PackageHelper(EGL ${EGL_FOUND} "${EGL_INCLUDE_DIRS}" "${EGL_LIBRARIES}")

#FFMPEG
if (SAIGA_WITH_FFMPEG)
    find_package(FFMPEG QUIET)
    if (FFMPEG_FOUND)
        SET(SAIGA_USE_FFMPEG 1)
    else ()
        SET(SAIGA_USE_FFMPEG 0)
    endif ()
    PackageHelper(FFMPEG ${FFMPEG_FOUND} "${FFMPEG_INCLUDE_DIR}" "${FFMPEG_LIBRARIES}")
endif ()


if (SAIGA_WITH_OPENVR)
    ## OpenVR / steamVR
    find_package(OpenVR QUIET)
    PackageHelper(OpenVR ${OPENVR_FOUND} "${OPENVR_INCLUDE_DIRS}" "${OPENVR_LIBRARY}")
    if (OPENVR_FOUND)
        set(SAIGA_VR 1)
    endif ()
endif ()

set(OPENGL_INCLUDES ${PACKAGE_INCLUDES})
set(OPENGL_LIBS ${LIBS})
set(OPENGL_TARGETS ${LIB_TARGETS})

if (SAIGA_USE_OPENGL)
    set(MODULE_OPENGL 1)
endif ()

