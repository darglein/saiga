/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "imageMetaData.h"

#include "saiga/core/math/math.h"

#include "internal/noGraphicsAPI.h"

namespace Saiga
{
std::ostream& operator<<(std::ostream& os, const ImageMetadata& d)
{
    os << "> ImageMetadata" << std::endl;
    os << "Size: " << d.width << "x" << d.height << std::endl;
    os << "DateTime: " << d.DateTime << std::endl;
    os << "Make: " << d.Make << std::endl;
    os << "Model: " << d.Model << std::endl;

    os << "FocalLengthMM: " << d.FocalLengthMM << std::endl;
    os << "FocalLengthMM35: " << d.FocalLengthMM35 << std::endl;

    std::string resStr;
    switch (d.FocalPlaneResolutionUnit)
    {
        case ImageMetadata::NoUnit:
            resStr = "NoUnit";
            break;
        case ImageMetadata::Inch:
            resStr = "Inch";
            break;
        case ImageMetadata::Centimeter:
            resStr = "Centimeter";
            break;
    }

    os << "FocalPlaneResolutionUnit: " << resStr << std::endl;
    os << "FocalPlaneXResolution: " << d.FocalPlaneXResolution << std::endl;
    os << "FocalPlaneYResolution: " << d.FocalPlaneYResolution;
    return os;
}

}  // namespace Saiga
