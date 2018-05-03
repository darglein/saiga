/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/util/glm.h"

namespace Saiga {


enum ImageChannel : int
{
    C1 = 0,C2,C3,C4
};

enum ImageElementType : int
{
    UCHAR = 0, SHORT, INT, FLOAT, DOUBLE, ELEMENT_UNKNOWN
};


static const int ImageElementTypeSize[] =
{
    1,2,4,4,8
};



enum ImageType : int
{
    UC1 = 0, UC2, UC3, UC4,
    S1, S2, S3, S4,
    I1, I2, I3, I4,
    F1, F2, F3, F4,
    D1, D2, D3, D4,
    TYPE_UNKNOWN
};

template<typename T>
struct SAIGA_GLOBAL ImageTypeTemplate{
};

template<> struct ImageTypeTemplate<unsigned char>{const static ImageType type = UC1;};
template<> struct ImageTypeTemplate<cvec2>{const static ImageType type = UC2;};
template<> struct ImageTypeTemplate<cvec3>{const static ImageType type = UC3;};
template<> struct ImageTypeTemplate<cvec4>{const static ImageType type = UC4;};

template<> struct ImageTypeTemplate<short>{const static ImageType type = S1;};
template<> struct ImageTypeTemplate<int>{const static ImageType type = I1;};
template<> struct ImageTypeTemplate<float>{const static ImageType type = F1;};



inline ImageType getType(ImageChannel channels, ImageElementType elementType)
{
    return ImageType(int(elementType) * 4 + int(channels));
}

inline ImageType getType(int channels, ImageElementType elementType)
{
    return ImageType(int(elementType) * 4 + int(channels-1));
}

inline int channels(ImageType type)
{
    return (int(type) % 4) + 1;
}

inline int elementType(ImageType type)
{
    return int(type) / 4;
}

inline int elementSize(ImageType type)
{
    return channels(type) * ImageElementTypeSize[elementType(type)];
}


}
