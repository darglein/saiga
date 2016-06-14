#pragma once

#include "saiga/text/textParameters.h"
#include "saiga/rendering/object3d.h"
#include "saiga/geometry/triangle_mesh.h"

#include <iostream>
#include <cstdint>

typedef std::vector<uint32_t> utf32string;

class SAIGA_GLOBAL Encoding {
public:

    static uint32_t UTF8toUTF32(const std::vector<unsigned char> &utf8char);
    static utf32string UTF8toUTF32(const std::string& str);

    static std::vector<unsigned char> UTF32toUTF8(uint32_t utf32char);
    static std::string UTF32toUTF8(const utf32string &str);
};
