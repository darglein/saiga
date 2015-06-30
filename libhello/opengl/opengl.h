#pragma once

#include <libhello/config.h>


#ifdef USE_GLEW
#include <GL/glew.h>
#endif

#ifdef USE_GLBINDING
#include <glbinding/gl/gl.h>
#include <glbinding/Binding.h>
using namespace gl;
#define GLFW_INCLUDE_NONE
#endif

extern void initOpenGL();
