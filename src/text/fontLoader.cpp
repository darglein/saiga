#include "saiga/text/fontLoader.h"
#include "saiga/text/textureAtlas.h"
#include "saiga/opengl/texture/texture.h"
#include "saiga/opengl/texture/textureLoader.h"
#include "saiga/util/assert.h"

FontLoader::FontLoader(const std::string &file)
    : file(file)
{

}

void FontLoader::loadMonochromatic(int fontSize)
{
    loadFace(fontSize);

    FT_Render_Mode renderMode = FT_RENDER_MODE_MONO; //This mode corresponds to 1-bit bitmaps (with 2 levels of opacity).
    FT_Error error;



    FT_ULong  charcode;
    FT_UInt   gindex;

    for(int i = 32; i < 128; i++) {
        int id = i-32;
        FT_UInt  glyph_index;

        /* retrieve glyph index from character code */
        glyph_index = FT_Get_Char_Index( face, i );

//        error = FT_Load_Glyph( face, glyph_index, FT_LOAD_DEFAULT );
                error = FT_Load_Glyph( face, glyph_index, FT_LOAD_TARGET_MONO );
        if ( error )
            continue;  /* ignore errors */

        addGlyph(i);
    }
//    charcode = FT_Get_First_Char( face, &gindex );
//    while ( gindex != 0 )
//    {
//        //... do something with (charcode,gindex) pair ...

//        if(charcode!=0){
//            error = FT_Load_Glyph( face, gindex, FT_LOAD_TARGET_MONO );

//            if(error==0){
//                addGlyph(charcode);
////                cout<<(char)charcode<<" "<<charcode<<" "<<gindex<<endl;
//            }
//        }


//        charcode = FT_Get_Next_Char( face, charcode, &gindex );
//    }

}

void FontLoader::addGlyph(int gindex)
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

    Image* image = new Image();
    myGlyph.bitmap = image;
    image->bitDepth = 8;
    image->channels = 1;
    image->width = source->width;
    image->height = source->rows;
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

            image->setPixel(x ,y,c);
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
    if(FT_New_Face(TextureAtlas::ft, file.c_str(), 0, &face)) {
        std::cerr<<"Could not open font "<<file<<std::endl;
        assert(0);
        return;
    }
    FT_Set_Pixel_Sizes(face, 0, fontSize);
}


void FontLoader::loadMonochromatic2(int fontSize)
{
    FT_Error error;
    FT_Render_Mode renderMode = FT_RENDER_MODE_NORMAL;
    if(FT_New_Face(TextureAtlas::ft, file.c_str(), 0, &face)) {
        std::cerr<<"Could not open font "<<file<<std::endl;
        assert(0);
        return;
    }
    FT_Set_Pixel_Sizes(face, 0, fontSize);



    const int count = 128-32;
    FT_Glyph glyphs[count], glyphs_bitmaps[count];
    FT_Glyph glyph_strokes[count], glyph_strokes_bitmaps[count];

    FT_GlyphSlot slot = (face)->glyph;
    for(int i = 32; i < 128; i++) {
        int id = i-32;
        FT_UInt  glyph_index;

        /* retrieve glyph index from character code */
        glyph_index = FT_Get_Char_Index( face, i );

        /* load glyph image into the slot (erase previous one) */
        //        error = FT_Load_Glyph( face, glyph_index, FT_LOAD_MONOCHROME );
        error = FT_Load_Glyph( face, glyph_index, FT_LOAD_DEFAULT );
        //        error = FT_Load_Glyph( face, glyph_index, FT_LOAD_TARGET_MONO );
        if ( error )
            continue;  /* ignore errors */

        error = FT_Get_Glyph( slot, &glyphs[id]);
        /* render the glyph to a bitmap, don't destroy original */
        glyphs_bitmaps[id] = glyphs[id];
        error = FT_Glyph_To_Bitmap( &glyphs_bitmaps[id], renderMode, NULL, 0 );

        FT_Glyph glyph = glyphs_bitmaps[id];


        if ( glyph->format != FT_GLYPH_FORMAT_BITMAP )
            cout<< "invalid glyph format returned!" <<endl;

        FT_BitmapGlyph bitmap = (FT_BitmapGlyph)glyph;
        FT_Bitmap* source = &bitmap->bitmap;



    }

    for(int i = 32; i < 128; i++) {

        FT_BitmapGlyph bitmap = (FT_BitmapGlyph)glyphs_bitmaps[i-32];
        FT_Bitmap* source = &bitmap->bitmap;


        Image* image = new Image();
        image->bitDepth = 8;
        image->channels = 1;
        image->width = source->width;
        image->height = source->rows;
        image->create();
        image->makeZero();


        for(int y = 0 ; y < source->rows  ; ++y){
            for(int x = 0 ; x < source->width ; ++x){
                unsigned char c;

                    c = source->buffer[y*(source->width) + x];

                uint16_t s = c;

                int ox = x + 0;
                int oy = y + 0;

                image->setPixel(ox ,oy,c);
            }
        }

        std::string str = "debug/fonts/"+std::to_string(i)+".png";
        if(!TextureLoader::instance()->saveImage(str,*image)){
            cout<<"could not save "<<str<<endl;
        }

    }

    for(int i = 32; i < 128; i++) {
        FT_Done_Glyph(glyphs[i-32]);
        FT_Done_Glyph(glyphs_bitmaps[i-32]);
    }
}

