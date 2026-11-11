/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/export.h"

// stb_image is compiled into saiga_core, so applications use this copy instead of bundling their own.
#ifndef STBIDEF
#    define STBIDEF extern SAIGA_CORE_API
#endif
#include "stb/stb_image.h"
