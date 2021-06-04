/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

// Include this file, if you want to use the OpenGL API.
// Do not directly include the loader library (glbinding/) or any other
// gl specific headers such as <gl.h>

#include "saiga/config.h"

#ifndef SAIGA_USE_OPENGL
#    error Saiga was build without opengl.
#endif

#include <glbinding/ProcAddress.h>
#include <glbinding/gl/gl.h>

// glbinding places all gl functions into the gl namespace.
using namespace gl;

// Make sure g nobody else includes gl.h after this.
#define __gl_h_
#define GLFW_INCLUDE_NONE

#define SAIGA_OPENGL_INCLUDED
