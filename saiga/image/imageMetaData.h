#pragma once

#include "saiga/config.h"

struct SAIGA_GLOBAL ImageMetadata{
    int width = 0;
    int height = 0;

    std::string DateTime;
    std::string Make;
    std::string Model;

    double FocalLengthMM = 0;
    double FocalLengthMM35 = 0;

    enum ResolutionUnit : int{
        NoUnit = 1,
        Inch = 2,
        Centimeter = 3
    };
    ResolutionUnit FocalPlaneResolutionUnit = NoUnit;
    double FocalPlaneXResolution = 0;
    double FocalPlaneYResolution = 0;

    SAIGA_GLOBAL friend std::ostream& operator<<(std::ostream& os, const ImageMetadata& d);
};
