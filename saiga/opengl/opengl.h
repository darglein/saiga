#pragma once

#include <saiga/config.h>
#include <string>

#ifdef USE_GLEW
#include <GL/glew.h>
typedef int MemoryBarrierMask;
#endif

#ifdef USE_GLBINDING
#include <glbinding/gl/gl.h>
#include <glbinding/Binding.h>
using namespace gl;
#define GLFW_INCLUDE_NONE
#endif

SAIGA_LOCAL extern void initOpenGL();
SAIGA_GLOBAL int getVersionMajor();
SAIGA_GLOBAL int getVersionMinor();

SAIGA_GLOBAL int getExtensionCount();
SAIGA_GLOBAL bool hasExtension(const std::string &ext);
