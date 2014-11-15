#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.
AC_PREREQ(2.61)
AC_INIT([libhello], [0.0.1], [darius.rueckert@fau.de])
AM_INIT_AUTOMAKE([subdir-objects])
LT_INIT
# name one of your source files here
#AC_CONFIG_SRCDIR([src/matrix.h])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_FILES(Makefile doc/Makefile doc/doxyfile doc/doxygen_mainpage.h)
AC_CONFIG_MACRO_DIR([m4])
# Checks for programs.
AC_PROG_CXX
AC_PROG_AWK
AC_PROG_CC
AC_PROG_CPP
AC_PROG_INSTALL
AC_PROG_LN_S
AC_PROG_MAKE_SET
AC_PROG_RANLIB
AX_CXX_COMPILE_STDCXX_11

# Checks for libraries.
AC_CHECK_LIB([png], [main],,exit)
AC_CHECK_LIB([m],[cos],,exit)
AC_CHECK_LIB([GLU], [gluLookAt],,exit)
AC_CHECK_LIB([glut], [glutInit],,exit)
AC_CHECK_LIB([SDL2],[SDL_Init],,exit)
AC_CHECK_LIB([GLEW],[glewInit],,exit)
AC_CHECK_LIB([GL],[glClear],,exit)
AC_CHECK_LIB([freetype],[FT_Init_FreeType],,exit)
#AC_CHECK_LIB([glfw3],[glfwInit],,exit)

PKG_CHECK_MODULES(GLFW,glfw3)

# Checks for header files.
AC_HEADER_STDC
AC_CHECK_HEADERS([inttypes.h stdlib.h string.h unistd.h])

#AC_CHECK_HEADER(glm/glm.hpp,[],AC_MSG_ERROR([Couldn't find glm/glm.hpp]))

AC_CHECK_HEADER(/usr/include/freetype2/config/ftheader.h,CPPFLAGS="$CPPFLAGS -I/usr/include/freetype2",AC_MSG_ERROR([Couldn't find config/ftheader.h]))
#AC_CHECK_HEADER(freetype2/config/ftheader.h,[],AC_MSG_ERROR([Couldn't find config/ftheader.h]))
AC_CHECK_HEADER(config/ftheader.h,[],AC_MSG_ERROR([Couldn't find config/ftheader.h]))

# Checks for typedefs, structures, and compiler characteristics.
AC_HEADER_STDBOOL
AC_C_INLINE
# Checks for library functions.
AC_CHECK_FUNCS([sqrt])




AC_OUTPUT
