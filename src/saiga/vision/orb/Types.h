#ifndef SAIGA_ORB_TYPES_H
#define SAIGA_ORB_TYPES_H

#include "saiga/core/image/imageView.h"
#include "saiga/core/image/templatedImage.h"
#include "saiga/vision/util/Features.h"

#include "ORB_config.h"

#ifdef ORB_USE_OPENCV
#    include <opencv2/core/types.hpp>
#endif

namespace Saiga
{
typedef unsigned char uchar;

typedef Saiga::ImageView<uchar> img_t;

typedef Saiga::KeyPoint<float> kpt_t;
}  // namespace SaigaORB
#endif  // SAIGA_ORB_TYPES_H
