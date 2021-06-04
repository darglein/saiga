/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/image/image.h"
#include "saiga/core/image/templatedImage.h"
#include "saiga/core/util/encoding.h"
#include "saiga/core/math/math.h"

#include <vector>

struct FT_FaceRec_;
struct FT_LibraryRec_;
typedef struct FT_FaceRec_* FT_Face;
typedef struct FT_LibraryRec_* FT_Library;

namespace Saiga
{
class FontLoader
{
   public:
    struct Glyph
    {
        int character;  // in ascii encoding
        vec2 advance;   // distance to the origin of the next character
        vec2 offset;    // offset of the bitmap position to the origin of this character
        vec2 size;      // size of bitmap
        TemplatedImage<unsigned char> bitmap;
    };

    std::vector<Glyph> glyphs;

    FontLoader(const std::string& file, const std::vector<Unicode::UnicodeBlock>& blocks = {Unicode::BasicLatin});
    ~FontLoader();
    void loadMonochromatic(int fontSize, int glyphPadding = 0);
    void writeGlyphsToFiles(const std::string& prefix);

   private:
    static FT_Library ft;
    std::string file;
    std::vector<Unicode::UnicodeBlock> blocks;
    FT_Face face = nullptr;

    void loadFace(int fontSize);
    void loadAndAddGlyph(int charCode, int glyphPadding);
    void addGlyph(int charCode, int glyphPadding);
};


}  // namespace Saiga
