/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "managedImage.h"

#include <png.h>


namespace Saiga
{
inline ImageType saigaType(int color_type, int bit_depth)
{
    int channels = -1;
    switch (color_type)
    {
        case PNG_COLOR_TYPE_GRAY:
            channels = 1;
            break;
        case PNG_COLOR_TYPE_GRAY_ALPHA:
            channels = 2;
            break;
        case PNG_COLOR_TYPE_RGB:
            channels = 3;
            break;
        case PNG_COLOR_TYPE_RGB_ALPHA:
            channels = 4;
            break;
        default:
            // Palette images are converted to RGB before this is called.
            return TYPE_UNKNOWN;
    }

    ImageElementType elementType = ImageElementType::IET_ELEMENT_UNKNOWN;
    switch (bit_depth)
    {
        case 8:
            elementType = ImageElementType::IET_UCHAR;
            break;
        case 16:
            elementType = ImageElementType::IET_USHORT;
            break;
        case 32:
            elementType = ImageElementType::IET_UINT;
            break;
        default:
            // 1, 2 and 4 bit images are expanded to 8 bit before this is called.
            return TYPE_UNKNOWN;
    }


    return getType(channels, elementType);
}

inline int PngColorType(ImageType saiga_type)
{
    int color_type = 0;

    switch (channels(saiga_type))
    {
        case 1:
            color_type = PNG_COLOR_TYPE_GRAY;
            break;
        case 2:
            color_type = PNG_COLOR_TYPE_GRAY_ALPHA;
            break;
        case 3:
            color_type = PNG_COLOR_TYPE_RGB;
            break;
        case 4:
            color_type = PNG_COLOR_TYPE_RGB_ALPHA;
            break;
        default:
            SAIGA_EXIT_ERROR("Invalid color type!");
    }
    return color_type;
}

}  // namespace Saiga


