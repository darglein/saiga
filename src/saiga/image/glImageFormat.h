/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/util/glm.h"
#include "saiga/opengl/opengl.h"
#include "saiga/image/imageFormat.h"

/**
 * Note:
 * This is separated from the normal imageFormat to get rid of the opengl dependency
 */

namespace Saiga {

SAIGA_GLOBAL GLenum getGlInternalFormat(ImageType type, bool srgb = false);
SAIGA_GLOBAL GLenum getGlFormat(ImageType type);
SAIGA_GLOBAL GLenum getGlType(ImageType type);

}
