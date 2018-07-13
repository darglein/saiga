/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/util/glm.h"
#include "saiga/image/imageView.h"


namespace Saiga {


void addAlphaChannel(ImageView<ucvec3> src, ImageView<ucvec4> dst, unsigned char alpha = 0);

}
