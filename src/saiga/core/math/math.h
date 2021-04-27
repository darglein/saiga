/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

// quat
#include "Quaternion.h"
// Basic Matrix and vector types such as vec3, mat4,...
#include "Types.h"
// A few GLM-like functions so we can write transpose(mat) instead of mat.transpose()
// This will most likely be removed in the future
#include "EigenGLMInterface.h"
#include "EigenGLMInterfaceFloat.h"
// Contains defintions for aligned allocations.
// This is important for eigen data structures
//#include "saiga/core/util/Align.h"
