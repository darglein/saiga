/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/image/imageFormat.h"
#include "saiga/opengl/opengl.h"
#include "saiga/core/util/math.h"

/**
 * Note:
 * This is separated from the normal imageFormat to get rid of the opengl dependency
 */

namespace Saiga
{
// TODO: Integer (not normalized) images for 8 and 16 bits.
SAIGA_GLOBAL GLenum getGlInternalFormat(ImageType type, bool srgb = false, bool integral = false);
SAIGA_GLOBAL GLenum getGlFormat(ImageType type);
SAIGA_GLOBAL GLenum getGlType(ImageType type);

}  // namespace Saiga
