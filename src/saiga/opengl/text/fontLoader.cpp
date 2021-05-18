/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "saiga/config.h"

#if defined(SAIGA_USE_FREETYPE)


#    include "saiga/core/util/assert.h"
#    include "saiga/core/util/fileChecker.h"
#    include "saiga/core/util/tostring.h"
#    include "saiga/opengl/text/fontLoader.h"
//#    include "saiga/opengl/text/textAtlas.h"
#    include "saiga/opengl/texture/Texture.h"
#    include "saiga/opengl/texture/TextureLoader.h"

#    include <freetype/ftstroke.h>

namespace Saiga
{
FT_Library FontLoader::ft = nullptr;

FontLoader::FontLoader(const std::string& _file, const std::vector<Unicode::UnicodeBlock>& blocks) : blocks(blocks)
{
    this->file = SearchPathes::font(_file);
    if (file == "")
    {
        std::cerr << "Could not open file " << _file << std::endl;
        std::cerr << SearchPathes::font << std::endl;
        SAIGA_ASSERT(0);
    }

    if (ft == nullptr)
    {
        if (FT_Init_FreeType(&ft))
        {
            std::cerr << "Could not init freetype library" << std::endl;
            SAIGA_ASSERT(0);
        }
    }
}

FontLoader::~FontLoader() {}

void FontLoader::loadMonochromatic(int fontSize, int glyphPadding)
{
    loadFace(fontSize);


    for (Unicode::UnicodeBlock block : blocks)
    {
        //        std::cout << "block " << block.start<< " " << block.end << std::endl;
        for (uint32_t i = block.start; i <= block.end; i++)
        {
            int charCode = i;
            loadAndAddGlyph(charCode, glyphPadding);
        }
    }

#    if 0
    for(Glyph& g : glyphs)
    {
        g.bitmap.save("debug/font/"+to_string(g.character)+"_2.png");
    }
#    endif
}

void FontLoader::loadAndAddGlyph(int charCode, int glyphPadding)
{
    //    std::cout << "loadAndAddGlyph "<<charCode << std::endl;
    FT_UInt glyph_index;

    /* retrieve glyph index from character code */
    glyph_index = FT_Get_Char_Index(face, charCode);

    // 0 glyph index means undefined character code
    if (glyph_index == 0)
    {
        //        std::cerr << "can't find glyph for charcode: " << std::hex << charCode << std::endl;
        return;
    }

    //        error = FT_Load_Glyph( face, glyph_index, FT_LOAD_DEFAULT );
    FT_Error error = FT_Load_Glyph(face, glyph_index, FT_LOAD_TARGET_MONO);
    if (error)
    {
        std::cerr << "random error" << std::endl;
        return; /* ignore errors */
    }

    addGlyph(charCode, glyphPadding);
}

void FontLoader::addGlyph(int charCode, int glyphPadding)
{
    //    if(charCode<32 || charCode>=256)
    //        return;
    FT_Error error;
    FT_Glyph glyph;
    FT_Glyph glyph_bitmap;

    FT_GlyphSlot slot = (face)->glyph;

    error = FT_Get_Glyph(slot, &glyph);
    SAIGA_ASSERT(error == 0);
    /* render the glyph to a bitmap, don't destroy original */
    glyph_bitmap = glyph;
    //    error = FT_Glyph_To_Bitmap( &glyph_bitmap, FT_RENDER_MODE_NORMAL, NULL, 0 );
    SAIGA_ASSERT(error == 0);
    error = FT_Glyph_To_Bitmap(&glyph_bitmap, FT_RENDER_MODE_MONO, NULL, 0);

    FT_Glyph g2 = glyph_bitmap;
    if (g2->format != FT_GLYPH_FORMAT_BITMAP)
    {
        std::cout << "invalid glyph format returned!" << std::endl;
        SAIGA_ASSERT(0);
    }

    FT_BitmapGlyph bitmap = (FT_BitmapGlyph)g2;
    FT_Bitmap* source     = &bitmap->bitmap;

    Glyph myGlyph;

    myGlyph.advance[0] = (g2->advance.x + 0x8000) >> 16;
    myGlyph.advance[1] = (g2->advance.y + 0x8000) >> 16;

    myGlyph.offset    = vec2(bitmap->left, bitmap->top);
    myGlyph.character = charCode;

    myGlyph.size = vec2(source->width + glyphPadding * 2, source->rows + glyphPadding * 2);


    myGlyph.bitmap.width  = source->width + glyphPadding * 2;
    myGlyph.bitmap.height = source->rows + glyphPadding * 2;
    myGlyph.bitmap.create();
    myGlyph.bitmap.makeZero();

    SAIGA_ASSERT(myGlyph.bitmap.type == UC1);


    for (int y = 0; y < (int)source->rows; ++y)
    {
        for (int x = 0; x < (int)source->width; ++x)
        {
            int byteIndex   = y * source->pitch + x / 8;
            int bitIndex    = 7 - (x % 8);
            unsigned char c = source->buffer[byteIndex];
            c               = (c >> bitIndex) & 0x1;
            if (c)
                c = 255;
            else
                c = 0;

            myGlyph.bitmap(y + glyphPadding, x + glyphPadding) = c;
        }
    }

    glyphs.push_back(myGlyph);

    FT_Done_Glyph(glyph);
    FT_Done_Glyph(glyph_bitmap);
}

void FontLoader::writeGlyphsToFiles(const std::string& prefix)
{
    for (Glyph& g : glyphs)
    {
        std::string str = prefix + std::to_string(g.character) + ".png";
        //        if(!TextureLoader::instance()->saveImage(str,*g.bitmap))
        if (g.bitmap.save(str))
        {
            std::cout << "could not save " << str << std::endl;
        }
    }
}



void FontLoader::loadFace(int fontSize)
{
    if (FT_New_Face(ft, file.c_str(), 0, &face))
    {
        std::cerr << "Could not open font " << file << std::endl;
        SAIGA_ASSERT(0);
        return;
    }
    FT_Set_Pixel_Sizes(face, 0, fontSize);
}

}  // namespace Saiga
#endif
