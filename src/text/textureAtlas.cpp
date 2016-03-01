#include "saiga/text/textureAtlas.h"
#include "saiga/opengl/texture/texture.h"
#include "saiga/opengl/texture/textureLoader.h"
#include "saiga/geometry/triangle_mesh.h"

#include <algorithm>
#include <FreeImagePlus.h>
#include <ft2build.h>
#include <ftstroke.h>
#include "saiga/util/assert.h"
#include FT_FREETYPE_H

#define NOMINMAX
#undef max
#undef min

FT_Library TextureAtlas::ft = nullptr;

TextureAtlas::TextureAtlas(){
    if(ft==nullptr){
        if(FT_Init_FreeType(&ft)) {
            std::cerr<< "Could not init freetype library"<<std::endl;
            assert(0);
        }
    }
}

TextureAtlas::~TextureAtlas()
{
    FT_Done_Face(face);
    delete textureAtlas;
}


void TextureAtlas::loadFont(const std::string &font, int font_size, int stroke_size){
    this->font = font;
    this->font_size = font_size;
    this->stroke_size = stroke_size;

    //    face = std::make_shared<FT_Face>();
    if(FT_New_Face(ft, font.c_str(), 0, &face)) {
        std::cerr<<"Could not open font "<<font<<std::endl;
        assert(0);
        return;
    }
    FT_Set_Pixel_Sizes(face, 0, font_size);

    createTextureAtlas();
}

void TextureAtlas::createTextureAtlas(){
    FT_Render_Mode renderMode = FT_RENDER_MODE_NORMAL; //it corresponds to 8-bit anti-aliased bitmaps.
//    FT_Render_Mode renderMode = FT_RENDER_MODE_MONO; //This mode corresponds to 1-bit bitmaps (with 2 levels of opacity).

    FT_Error error;
    FT_Stroker stroker = 0;

    if(stroke_size>0){
        // Set up a stroker.
        FT_Stroker_New(ft, &stroker);
        FT_Stroker_Set(stroker,
                       stroke_size,
                       FT_STROKER_LINECAP_ROUND,
                       FT_STROKER_LINEJOIN_ROUND,
                       0);
    }


//    charPaddingX = 4;
//    charPaddingY = 4;
//    charBorder = 4;

    int chars= 0;
    int w=0,h=0;


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

        if(stroke_size>0){
            error = FT_Get_Glyph( slot, &glyph_strokes[id] );
            error = FT_Glyph_Stroke( &glyph_strokes[id], stroker, 1 );

            glyph_strokes_bitmaps[id] = glyph_strokes[id];
            error = FT_Glyph_To_Bitmap( &glyph_strokes_bitmaps[id], renderMode, NULL, 0 );
            glyph = glyph_strokes_bitmaps[id];
        }





        if ( glyph->format != FT_GLYPH_FORMAT_BITMAP )
            cout<< "invalid glyph format returned!" <<endl;

        FT_BitmapGlyph bitmap = (FT_BitmapGlyph)glyph;
        FT_Bitmap* source = &bitmap->bitmap;

        character_info &info = characters[i];

        info.ax = ( glyph->advance.x + 0x8000 ) >> 16;
        info.ay = ( glyph->advance.y + 0x8000 ) >> 16;
        info.ax += stroke_size/64; //????

        info.bw = source->width;
        info.bh = source->rows;

        info.bl = bitmap->left;
        info.bt = bitmap->top;


        maxCharacter.min = glm::min(maxCharacter.min,vec3(info.bl,info.bt-info.bh,0));
        maxCharacter.max = glm::max(maxCharacter.max,vec3(info.bl+info.bw,info.bt-info.bh+info.bh,0));

        chars++;


    }




    int charsPerRow = glm::ceil(glm::sqrt((float)chars));
    //    cout<<"chars "<<chars<<" charsperrow "<<charsPerRow<<" total "<<charsPerRow*charsPerRow<<endl;

    atlasHeight = charPaddingY;
    atlasWidth = 0;

    for(int cy = 0 ; cy < charsPerRow ; ++cy){
        int currentW = charPaddingX;
        int currentH = charPaddingY;

        for(int cx = 0 ; cx < charsPerRow ; ++cx){
            int i = cy * charsPerRow + cx;
            if(i>=chars)
                break;
            character_info &info = characters[i+32];

            info.atlasX = currentW;
            info.atlasY = atlasHeight;

            currentW += info.bw+charPaddingX;
            currentH = std::max(currentH, info.bh);
        }
        atlasWidth = std::max(currentW, atlasWidth);
        atlasHeight += currentH+charPaddingY;
    }

    //increase width to a number dividable by 8 to fix possible alignment issues
    while(atlasWidth%8!=0)
        atlasWidth++;

    h = atlasHeight;
    w = atlasWidth;

    cout<<"AtlasWidth "<<w<<" AtlasHeight "<<h<<endl;

    Image img;
    img.bitDepth = 8;
    img.channels = 2;
    img.width = w;
    img.height = h;
    img.create();
    img.makeZero();


//    img.addChannel();



    for(int i = 32; i < 128; i++) {

        FT_BitmapGlyph bitmap = (FT_BitmapGlyph)glyphs_bitmaps[i-32];
        FT_Bitmap* source = &bitmap->bitmap;
        cout<<"FT_Bitmap "<<source->width<<","<<source->rows<<endl;

        character_info &info = characters[i];


        float tx = (float)info.atlasX / (float)w;
        float ty = (float)info.atlasY / (float)h;

        info.tcMin = vec2(tx,ty);
        info.tcMax = vec2(tx+(float)info.bw/(float)w,ty+(float)info.bh/(float)h);

        //offset from normal glyph relative to stroke glyph
        int offsetX=0,offsetY=0;
        if(stroke_size>0){
            FT_BitmapGlyph bitmapstroke = (FT_BitmapGlyph)glyph_strokes_bitmaps[i-32];
            FT_Bitmap* sourceStroke = &bitmapstroke->bitmap;
            offsetX = bitmap->left - bitmapstroke->left;
            offsetY = -bitmap->top + bitmapstroke->top;

            //        cout<<offsetX<<" "<<offsetY<<endl;

            for(int y = 0 ; y < info.bh  ; ++y){
                for(int x = 0 ; x < info.bw ; ++x){
                    unsigned char c = sourceStroke->buffer[y*(info.bw) + x];
                    uint16_t s = c;
                    img.setPixel(info.atlasX+x ,info.atlasY+y,s);
                }
            }
        }

        for(int y = 0 ; y < source->rows  ; ++y){
            for(int x = 0 ; x < source->width ; ++x){
                unsigned char c;
                if(renderMode==FT_RENDER_MODE_MONO){
                    int byteIndex = y*source->pitch + x/8;
                    int bitIndex = 7 - (x % 8);
                    unsigned char c = source->buffer[byteIndex];
                    c = (c>>bitIndex) & 0x1;
                    if(c)
                        c = 255;
                }else{
                    c = source->buffer[y*(source->width) + x];
                }

                uint16_t s = c;

                int ox = x + offsetX;
                int oy = y + offsetY;

                uint16_t old = img.getPixel<uint16_t>(info.atlasX+ox ,info.atlasY+oy);
                old = old + (s<<8);
                img.setPixel(info.atlasX+ox ,info.atlasY+oy,old);
            }
        }

    }


    //cleanup freetype stuff
    if(stroke_size>0){
        FT_Stroker_Done(stroker);
        for(int i = 32; i < 128; i++) {
            FT_Done_Glyph(glyph_strokes[i-32]);
            FT_Done_Glyph(glyph_strokes_bitmaps[i-32]);
        }
    }
    for(int i = 32; i < 128; i++) {
        FT_Done_Glyph(glyphs[i-32]);
        FT_Done_Glyph(glyphs_bitmaps[i-32]);
    }
    img.addChannel();


//    fipImage fipimage;
//    img.convertTo(fipimage);

//    if(!fipimage.save(str.c_str())){
//        cout<<"could not save "<<str<<endl;
//    }
    std::string str = "debug/test"+std::to_string(w)+".png";
    if(!TextureLoader::instance()->saveImage(str,img)){
        cout<<"could not save "<<str<<endl;
    }

    //    std::vector<GLubyte> data(w*h,0x00);
    textureAtlas = new Texture();

    //zero initialize texture
    //        textureAtlas->createTexture(w ,h,GL_RED, GL_R8  ,GL_UNSIGNED_BYTE,img.data);
    //    textureAtlas->createTexture(w ,h,GL_RED, GL_R8  ,GL_UNSIGNED_BYTE,0);

    textureAtlas->fromImage(img);

    textureAtlas->bind();
    // The allowable values are 1 (byte-alignment), 2 (rows aligned to even-numbered bytes), Default: 4 (word-alignment), and 8 (rows start on double-word boundaries)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    textureAtlas->unbind();

    //create mipmaps to reduce aliasing effects
    textureAtlas->generateMipmaps();
}

