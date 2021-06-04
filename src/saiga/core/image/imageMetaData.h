/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

#include <iostream>
namespace Saiga
{
struct SAIGA_CORE_API ImageMetadata
{
    int width  = 0;
    int height = 0;

    std::string DateTime;
    std::string Make;
    std::string Model;

    double FocalLengthMM   = 0;
    double FocalLengthMM35 = 0;

    enum ResolutionUnit : int
    {
        NoUnit     = 1,
        Inch       = 2,
        Centimeter = 3
    };
    ResolutionUnit FocalPlaneResolutionUnit = NoUnit;
    double FocalPlaneXResolution            = 0;
    double FocalPlaneYResolution            = 0;
    double ExposureTime                     = 0;
    double ISOSpeedRatings                  = 0;

    SAIGA_CORE_API friend std::ostream& operator<<(std::ostream& os, const ImageMetadata& d);
};

}  // namespace Saiga
