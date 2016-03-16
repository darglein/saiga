#include "saiga/text/fontLoader.h"
#include "saiga/text/textureAtlas.h"
#include "saiga/opengl/texture/texture.h"
#include "saiga/opengl/texture/textureLoader.h"
#include "saiga/util/assert.h"

#include <ft2build.h>
#include <ftstroke.h>

FT_Library FontLoader::ft = nullptr;

FontLoader::FontLoader(const std::string &file)
    : file(file)
{
    if(ft==nullptr){
        if(FT_Init_FreeType(&ft)) {
            std::cerr<< "Could not init freetype library"<<std::endl;
            assert(0);
        }
    }
}

FontLoader::~FontLoader()
{
    //TODO shared pointer
    for(Glyph &g : glyphs){
        delete g.bitmap;
    }


}

void FontLoader::loadMonochromatic(int fontSize, int glyphPadding)
{
    loadFace(fontSize);

    FT_Error error;

    for(int i = 32; i < 128; i++) {
        FT_UInt  glyph_index;

        /* retrieve glyph index from character code */
        glyph_index = FT_Get_Char_Index( face, i );

//        error = FT_Load_Glyph( face, glyph_index, FT_LOAD_DEFAULT );
                error = FT_Load_Glyph( face, glyph_index, FT_LOAD_TARGET_MONO );
        if ( error )
            continue;  /* ignore errors */

        addGlyph(i,glyphPadding);
    }

}

void FontLoader::addGlyph(int gindex, int glyphPadding)
{
    if(gindex<32 || gindex>127)
        return;
    FT_Error error;
    FT_Glyph glyph;
    FT_Glyph glyph_bitmap;

    FT_GlyphSlot slot = (face)->glyph;

    error = FT_Get_Glyph( slot, &glyph);
    assert(error==0);
    /* render the glyph to a bitmap, don't destroy original */
    glyph_bitmap = glyph;
//    error = FT_Glyph_To_Bitmap( &glyph_bitmap, FT_RENDER_MODE_NORMAL, NULL, 0 );
    assert(error==0);
    error = FT_Glyph_To_Bitmap( &glyph_bitmap, FT_RENDER_MODE_MONO, NULL, 0 );

    FT_Glyph g2 = glyph_bitmap;
    if ( g2->format != FT_GLYPH_FORMAT_BITMAP ){
        cout<< "invalid glyph format returned!" <<endl;
        assert(0);
    }

    FT_BitmapGlyph bitmap = (FT_BitmapGlyph)g2;
    FT_Bitmap* source = &bitmap->bitmap;

    Glyph myGlyph;

    myGlyph.advance.x = ( g2->advance.x + 0x8000 ) >> 16;
    myGlyph.advance.y = ( g2->advance.y + 0x8000 ) >> 16;

    myGlyph.offset = vec2(bitmap->left,bitmap->top);
    myGlyph.character = gindex;

    myGlyph.size = vec2(source->width+glyphPadding*2,source->rows+glyphPadding*2);

    Image* image = new Image();
    myGlyph.bitmap = image;
    image->bitDepth = 8;
    image->channels = 1;
    image->width = source->width+glyphPadding*2;
    image->height = source->rows+glyphPadding*2;
    image->create();
    image->makeZero();

    glyphs.push_back(myGlyph);

    for(int y = 0 ; y < source->rows  ; ++y){
        for(int x = 0 ; x < source->width ; ++x){
            int byteIndex = y*source->pitch + x/8;
            int bitIndex = 7 - (x % 8);
            unsigned char c = source->buffer[byteIndex];
            c = (c>>bitIndex) & 0x1;
            if(c)
                c = 255;

            image->setPixel(x+glyphPadding ,y+glyphPadding,c);
        }
    }

    FT_Done_Glyph(glyph);
    FT_Done_Glyph(glyph_bitmap);
}

void FontLoader::writeGlyphsToFiles(const std::string &prefix)
{
    for(Glyph& g : glyphs){
        std::string str = prefix+std::to_string(g.character)+".png";
        if(!TextureLoader::instance()->saveImage(str,*g.bitmap)){
            cout<<"could not save "<<str<<endl;
        }

    }
}



void FontLoader::loadFace(int fontSize)
{
    if(FT_New_Face(ft, file.c_str(), 0, &face)) {
        std::cerr<<"Could not open font "<<file<<std::endl;
        assert(0);
        return;
    }
    FT_Set_Pixel_Sizes(face, 0, fontSize);
}



