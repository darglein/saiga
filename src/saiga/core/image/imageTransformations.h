/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/math.h"

#include "templatedImage.h"


namespace Saiga
{
namespace ImageTransformation
{
SAIGA_CORE_API void addAlphaChannel(ImageView<const ucvec3> src, ImageView<ucvec4> dst, unsigned char alpha = 255);
SAIGA_CORE_API void RemoveAlphaChannel(ImageView<const ucvec4> src, ImageView<ucvec3> dst);

// depth to rgb image only for visualizaion
SAIGA_CORE_API void depthToRGBA(ImageView<const uint16_t> src, ImageView<ucvec4> dst, uint16_t minD, uint16_t maxD);
SAIGA_CORE_API void depthToRGBA(ImageView<const float> src, ImageView<ucvec4> dst, float minD = 0, float maxD = 7);
SAIGA_CORE_API void depthToRGBA_HSV(ImageView<const float> src, ImageView<ucvec4> dst, float minD = 0, float maxD = 7);

SAIGA_CORE_API void RGBAToGray8(ImageView<const ucvec4> src, ImageView<unsigned char> dst);
// with scale = 1 the resulting grayscale will be in the range 0..255
SAIGA_CORE_API void RGBAToGrayF(ImageView<const ucvec4> src, ImageView<float> dst, float scale = 1.0f);

// just sets the color channels to the gray value
SAIGA_CORE_API void Gray8ToRGBA(ImageView<unsigned char> src, ImageView<ucvec4> dst, unsigned char alpha = 255);
SAIGA_CORE_API void Gray8ToRGB(ImageView<unsigned char> src, ImageView<ucvec3> dst);



SAIGA_CORE_API void ScaleDown2(ImageView<const ucvec4> src, ImageView<ucvec4> dst);


SAIGA_CORE_API float sharpness(ImageView<const unsigned char> src);
/**
 * Converts a floating point image to a 8-bit image and saves it.
 * Useful for debugging.
 */
SAIGA_CORE_API bool saveHSV(const std::string& path, ImageView<float> img, float vmin, float vmax);
SAIGA_CORE_API bool save(const std::string& path, ImageView<float> img, float vmin, float vmax);

// Computes the per pixel error and writes it into an output image.
// The error is computed as (img1 - img2).abs().maxChannel()
//
SAIGA_CORE_API TemplatedImage<unsigned char> AbsolutePixelError(ImageView<const ucvec3> img1,
                                                                ImageView<const ucvec3> img2);
SAIGA_CORE_API TemplatedImage<unsigned char> AbsolutePixelError(ImageView<const unsigned char> img1,
                                                                ImageView<const unsigned char> img2);


// Per pixel error colorized using the turbo color bar.
SAIGA_CORE_API TemplatedImage<ucvec3> ErrorImage(ImageView<ucvec3> img1, ImageView<ucvec3> img2);
SAIGA_CORE_API TemplatedImage<ucvec3> ErrorImage(ImageView<unsigned char> img1, ImageView<unsigned char> img2);

// Absolute image difference summed up over all pixels (will get very large)
SAIGA_CORE_API long L1Difference(ImageView<const ucvec3> img1, ImageView<const ucvec3> img2);
SAIGA_CORE_API long L1Difference(ImageView<const unsigned char> img1, ImageView<const unsigned char> img2);

}  // namespace ImageTransformation
}  // namespace Saiga
