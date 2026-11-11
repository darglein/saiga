/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/export.h"

// stb_image_write is compiled into saiga_core, so applications use this copy instead of bundling their own.
#ifndef STBIWDEF
#    define STBIWDEF extern SAIGA_CORE_API
#endif
#include "stb/stb_image_write.h"
