/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "encoding.h"

#include "saiga/core/util/assert.h"

#include "internal/noGraphicsAPI.h"

#include <iostream>

namespace Saiga
{
int sizeTable[] = {
    // ascii characters: starting with 0
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,

    // invalid characters: starting with 10
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,

    // 2 bytes characters: starting with 110
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,

    // 3 bytes characters: starting with 1110
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,

    // 4 bytes characters: starting with 11110
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,

    // invalid characters starting with 11111
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
};


uint32_t Encoding::UTF8toUTF32(const std::vector<unsigned char>& utf8char)
{
    int size = utf8char.size();
    SAIGA_ASSERT(size >= 1 && size <= 4);

    uint32_t result = 0;
    switch (size)
    {
        case 1:
            // format: 1|7
            result = utf8char[0] & 0x7F;
            break;
        case 2:
            // format 3|5 2|6
            result = ((utf8char[0] & 0x1F) << 6) | ((utf8char[1] & 0x3F) << 0);
            break;
        case 3:
            // format 4|4 2|6 2|6
            result = ((utf8char[0] & 0xF) << 12) | ((utf8char[1] & 0x3F) << 6) | ((utf8char[2] & 0x3F) << 0);
            break;
        case 4:
            // format 5|3 2|6 2|6 2|6
            result = ((utf8char[0] & 0x7) << 18) | ((utf8char[1] & 0x3F) << 12) | ((utf8char[2] & 0x3F) << 6) |
                     ((utf8char[3] & 0x3F) << 0);
            break;
    }
    return result;
}

utf32string Encoding::UTF8toUTF32(const std::string& str)
{
    utf32string result;

    for (int i = 0; i < (int)str.size();)
    {
        unsigned char c = str[i];
        int size        = sizeTable[c];

        std::vector<unsigned char> utf8char;
        for (int j = 0; j < size; ++j)
        {
            utf8char.push_back(str[i + j]);
        }

        // check if the following bytes start with 01
        for (int j = 1; j < size; ++j)
        {
            unsigned char c = utf8char[j];
            if ((c >> 6) != 0x2)
            {
                size = -1;
            }
        }

        if (size == -1)
        {
            std::cerr << "Warning Encoding::UTF8toUTF32: The passed string is not UTF8 encoded! " << str << std::endl;
            size = 1;
        }
        else
        {
            result.push_back(Encoding::UTF8toUTF32(utf8char));
        }
        i += size;
    }

    return result;
}

std::vector<unsigned char> Encoding::UTF32toUTF8(uint32_t utf32char)
{
    std::vector<unsigned char> result;

    int size = 0;


    if (utf32char <= 0x7F)
        size = 1;
    else if (utf32char >= 0x80 && utf32char <= 0x7FF)
        size = 2;
    else if (utf32char >= 800 && utf32char <= 0xFFFF)
        size = 3;
    else if (utf32char >= 10000 && utf32char <= 0x1FFFFF)
        size = 4;
    else
    {
        // this utf32 character is not convertible to utf8
        size = 0;
    }

    switch (size)
    {
        case 1:
            // format: 1|7
            utf32char = utf32char & 0x7F;
            result.push_back(utf32char);
            break;
        case 2:
            // format 3|5 2|6
            utf32char = utf32char & 0x7FF;
            result.push_back((utf32char >> 6) | (0x6 << 5));
            result.push_back(((utf32char >> 0) & 0x3F) | (0x2 << 6));
            break;
        case 3:
            // format 4|4 2|6 2|6
            utf32char = utf32char & 0xFFFF;
            result.push_back((utf32char >> 12) | (0x6 << 5));
            result.push_back(((utf32char >> 6) & 0x3F) | (0x2 << 6));
            result.push_back(((utf32char >> 0) & 0x3F) | (0x2 << 6));
            break;
        case 4:
            // format 5|3 2|6 2|6 2|6
            utf32char = utf32char & 0x1FFFFF;
            result.push_back((utf32char >> 18) | (0x6 << 5));
            result.push_back(((utf32char >> 12) & 0x3F) | (0x2 << 6));
            result.push_back(((utf32char >> 6) & 0x3F) | (0x2 << 6));
            result.push_back(((utf32char >> 0) & 0x3F) | (0x2 << 6));
            break;
    }
    return result;
}

std::string Encoding::UTF32toUTF8(const utf32string& str)
{
    std::string result;

    for (uint32_t c : str)
    {
        std::vector<unsigned char> utf8char = Encoding::UTF32toUTF8(c);
        for (unsigned char uc : utf8char)
        {
            result.push_back(uc);
        }
    }
    return result;
}

}  // namespace Saiga
