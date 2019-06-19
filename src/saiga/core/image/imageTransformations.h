/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/math.h"

#include "imageView.h"


namespace Saiga
{
namespace ImageTransformation
{
SAIGA_CORE_API void addAlphaChannel(ImageView<const ucvec3> src, ImageView<ucvec4> dst, unsigned char alpha = 255);

// depth to rgb image only for visualizaion
SAIGA_CORE_API void depthToRGBA(ImageView<const uint16_t> src, ImageView<ucvec4> dst, uint16_t minD, uint16_t maxD);
SAIGA_CORE_API void depthToRGBA(ImageView<const float> src, ImageView<ucvec4> dst, float minD = 0, float maxD = 7);

SAIGA_CORE_API void RGBAToGray8(ImageView<const ucvec4> src, ImageView<unsigned char> dst);
// with scale = 1 the resulting grayscale will be in the range 0..255
SAIGA_CORE_API void RGBAToGrayF(ImageView<const ucvec4> src, ImageView<float> dst, float scale = 1.0f);


SAIGA_CORE_API float sharpness(ImageView<const unsigned char> src);
/**
 * Converts a floating point image to a 8-bit image and saves it.
 * Useful for debugging.
 */
SAIGA_CORE_API bool saveHSV(const std::string& path, ImageView<float> img, float vmin, float vmax);
SAIGA_CORE_API bool save(const std::string& path, ImageView<float> img, float vmin, float vmax);

}  // namespace ImageTransformation
}  // namespace Saiga
