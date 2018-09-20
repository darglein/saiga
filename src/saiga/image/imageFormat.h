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
    CHANNEL_1 = 0,
    CHANNEL_2,
    CHANNEL_3,
    CHANNEL_4
};

enum ImageElementType : int
{
    IET_CHAR = 0, IET_UCHAR,
    IET_SHORT, IET_USHORT,
    IET_INT, IET_UINT,
    IET_FLOAT, IET_DOUBLE,
    IET_ELEMENT_UNKNOWN
};


static const int ImageElementTypeSize[] =
{
    1,1,
    2,2,
    4,4,
    4,8,
    0
};



enum ImageType : int
{
    C1 = 0, C2, C3, C4,
    UC1, UC2, UC3, UC4,

    S1, S2, S3, S4,
    US1, US2, US3, US4,

    I1, I2, I3, I4,
    UI1, UI2, UI3, UI4,

    F1, F2, F3, F4,
    D1, D2, D3, D4,

    TYPE_UNKNOWN
};

template<typename T>
struct SAIGA_GLOBAL ImageTypeTemplate{
};

template<> struct ImageTypeTemplate<char>{
    using ChannelType = char;
    const static ImageType type = C1;
};
template<> struct ImageTypeTemplate<cvec2>{
    using ChannelType = char;
    const static ImageType type = C2;
};
template<> struct ImageTypeTemplate<cvec3>{
    using ChannelType = char;
    const static ImageType type = C3;
};
template<> struct ImageTypeTemplate<cvec4>{
    using ChannelType = char;
    const static ImageType type = C4;
};

template<> struct ImageTypeTemplate<unsigned char>{
    using ChannelType = unsigned char;
    const static ImageType type = UC1;
};
template<> struct ImageTypeTemplate<ucvec2>{
    using ChannelType = unsigned char;
    const static ImageType type = UC2;
};
template<> struct ImageTypeTemplate<ucvec3>{
    using ChannelType = unsigned char;
    const static ImageType type = UC3;
};
template<> struct ImageTypeTemplate<ucvec4>{
    using ChannelType = unsigned char;
    const static ImageType type = UC4;
};

template<> struct ImageTypeTemplate<short>{
    using ChannelType = short;
    const static ImageType type = S1;
};
template<> struct ImageTypeTemplate<unsigned short>{
    using ChannelType = unsigned short;
    const static ImageType type = US1;
};

template<> struct ImageTypeTemplate<int>{
    using ChannelType = int;
    const static ImageType type = I1;
};
template<> struct ImageTypeTemplate<unsigned int>{
    using ChannelType = unsigned int;
    const static ImageType type = UI1;
};

template<> struct ImageTypeTemplate<float>{
    using ChannelType = float;
    const static ImageType type = F1;
};



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
