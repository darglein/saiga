/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#ifndef SHADER_CONFIG_H
#define SHADER_CONFIG_H


#if defined(GL_core_profile)
#define SHADER_DEVICE
#else
#define SHADER_HOST
#endif


#ifdef SHADER_HOST
#define FUNC_DECL inline
#include "saiga/util/glm.h"
using namespace glm;
#else
#define FUNC_DECL
#endif



#endif
