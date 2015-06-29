#include "text/text_generator.h"
#include "libhello/opengl/texture/texture.h"
#include <algorithm>
#include <FreeImagePlus.h>
#include <ft2build.h>
#include <freetype2/freetype/ftstroke.h>
#include FT_FREETYPE_H

#define NOMINMAX
#undef max
#undef min

FT_Library TextGenerator::ft = nullptr;

TextGenerator::TextGenerator(){

    if(ft==nullptr){
//        ft = FT_Library> (new FT_Library());
        if(FT_Init_FreeType(&ft)) {
            cerr<< "Could not init freetype library"<<endl;
            exit(1);
        }
    }
}

TextGenerator::~TextGenerator()
{
    FT_Done_Face(face);
    delete textureAtlas;
}


void TextGenerator::loadFont(const std::string &font, int font_size){
    this->font = font;
    this->font_size = font_size;

//    face = std::make_shared<FT_Face>();
    if(FT_New_Face(ft, font.c_str(), 0, &face)) {
        cerr<<"Could not open font "<<font<<endl;
//        assert(0);
        return;
    }
    FT_Set_Pixel_Sizes(face, 0, font_size);

    createTextureAtlas();
}

void TextGenerator::createTextureAtlas(){
    charPaddingX = 2;
    charPaddingY = 2;
    charBorder = 0;

    int chars= 0;
    int w=0,h=0;
    FT_GlyphSlot g = (face)->glyph;
    for(int i = 32; i < 128; i++) {
        if(FT_Load_Char(face, i, FT_LOAD_RENDER)) {
            cerr<<"Could not load character '"<<(char)i<<"'"<<endl;
            continue;
        }



        character_info &info = characters[i];

        info.ax = (g->advance.x >> 6) + 2*charBorder;
        info.ay = g->advance.y >> 6;

        info.bw = g->bitmap.width + 2*charBorder;
        info.bh = g->bitmap.rows + 2*charBorder;

        info.bl = g->bitmap_left;
        info.bt = g->bitmap_top;


        maxCharacter.min = glm::min(maxCharacter.min,vec3(info.bl,info.bt-info.bh,0));
        maxCharacter.max = glm::max(maxCharacter.max,vec3(info.bl+info.bw,info.bt-info.bh+info.bh,0));

        //all characters in one row
//        info.atlasX = w;
//        info.atlasY = 0;
//        w += info.bw+charOffset;
//        h = std::max(h, (int)info.bh);

        chars++;

    }

    int charsPerRow = glm::ceil(glm::sqrt((float)chars));
//    cout<<"chars "<<chars<<" charsperrow "<<charsPerRow<<" total "<<charsPerRow*charsPerRow<<endl;

    int atlasHeight = 0;
    int atlasWidth = 0;

    for(int cy = 0 ; cy < charsPerRow ; ++cy){
        int currentW = 0;
        int currentH = 0;

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

    cout<<"AtlasWidth "<<atlasWidth<<" AtlasHeight "<<atlasHeight<<endl;

    h = atlasHeight;
    w = atlasWidth;

    //increase width to a number dividable by 8 to fix possible alignment issues
    while(w%8!=0)
        w++;

    Image img;
    img.bitDepth = 8;
    img.channels = 1;
    img.width = w;
    img.height = h;
    img.create();
    img.makeZero();


    img.addChannel();

    // Set up a stroker.
    FT_Stroker stroker;
    FT_Stroker_New(ft, &stroker);
    FT_Stroker_Set(stroker,
                   (int)( 64),
                   FT_STROKER_LINECAP_ROUND,
                   FT_STROKER_LINEJOIN_ROUND,
                   0);



    for(int i = 32; i < 128; i++) {
        if(FT_Load_Char(face, i, FT_LOAD_RENDER | FT_LOAD_NO_BITMAP))
            continue;

        FT_Glyph  glyph;
        FT_Get_Glyph( g, &glyph );

        character_info &info = characters[i];


        float tx = (float)info.atlasX / (float)w;
        float ty = (float)info.atlasY / (float)h;

        info.tcMin = vec2(tx,ty);
        info.tcMax = vec2(tx+(float)info.bw/(float)w,ty+(float)info.bh/(float)h);

//        textureAtlas->uploadSubImage(info.atlasX, info.atlasY, info.bw, info.bh, g->bitmap.buffer);
//        img.setSubImage(info.atlasX, info.atlasY, info.bw, info.bh, g->bitmap.buffer);

        //place character int the middle of the block
        for(int y = 0 ; y < info.bh - 2*charBorder ; ++y){
            for(int x = 0 ; x < info.bw - 2*charBorder ; ++x){
                unsigned char c = g->bitmap.buffer[y*(info.bw- 2*charBorder) + x];
                uint16_t s = c;
                img.setPixel(info.atlasX+x + charBorder,info.atlasY+y+ charBorder,s);
            }
        }

        FT_Done_Glyph( glyph );

//        //create border
//        for(int y = 0 ; y < info.bh; ++y){
//            for(int x = 0 ; x < info.bw; ++x){

////                uint16_t c = img.getPixel<uint16_t>(info.atlasX+x,info.atlasY+y);
////                if(c<<8 == 0xFF00)
////                    continue;

////                for(int dy = -charBorder; dy<=charBorder;++dy){
////                    for(int dx = -charBorder; dx<=charBorder;++dx){
////                        int nx = x + dx;
////                        int ny = y + dy;
////                        if(nx<0 || nx >=info.bw || ny < 0 || ny >=info.bh)
////                            continue;

////                         uint16_t d = img.getPixel<uint16_t>(info.atlasX+nx,info.atlasY+ny);
////                         if(d<<8 == 0xFF00){
////                             c = c | 0xFF00;
//////                             cout<<"found"<<endl;
////                         }
////                    }
////                }
////                img.setPixel(info.atlasX+x,info.atlasY+y,c);




//                //sobel
//                int weights[3][3] = { {-1,-2,-1},
//                                 {0,0,0},
//                                 {1,2,1}};

//                int sum = 0;
//                for(int dy = -1 ; dy <= 1 ; ++dy){
//                    for(int dx = -1 ; dx <= 1 ; ++dx){
//                        int nx = x + dx;
//                        int ny = y + dy;
//                        if(nx<0 || nx >=info.bw || ny < 0 || ny >=info.bh)
//                            continue;
//                        int weight = weights[dx+1][dy+1];
//                        uint16_t d = img.getPixel<uint16_t>(info.atlasX+nx,info.atlasY+ny);
//                        sum += (d & 0x00FF) * weight;
//                    }
//                }
//                sum = glm::abs(sum);

//                uint16_t c = img.getPixel<uint16_t>(info.atlasX+x,info.atlasY+y);
//                c = c | ((sum & 0x00FF)<<8);


//                img.setPixel(info.atlasX+x,info.atlasY+y,c);

//            }
//        }
    }

    FT_Stroker_Done(stroker);

    img.addChannel();


    fipImage fipimage;
    img.convertTo(fipimage);
    std::string str = "debug/test"+std::to_string(w)+".png";
    if(!fipimage.save(str.c_str())){
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
}



void TextGenerator::createTextMesh(TriangleMesh<VertexNT, GLuint> &mesh, const std::string &text, int startX, int startY){

    int x=startX,y=startY;
    VertexNT verts[4];
    for(char c : text){
        character_info &info = characters[(int)c];

        vec3 offset = vec3(x+info.bl,y+info.bt-info.bh,0);


        //bottom left
        verts[0] = VertexNT(offset,
                            vec3(0,0,1),
                            vec2(info.tcMin.x,info.tcMax.y));
        //bottom right
        verts[1] = VertexNT(offset+vec3(info.bw,0,0),
                            vec3(0,0,1),
                            vec2(info.tcMax.x,info.tcMax.y));
        //top right
        verts[2] = VertexNT(offset+vec3(info.bw,info.bh,0),
                            vec3(0,0,1),
                            vec2(info.tcMax.x,info.tcMin.y));
        //top left
        verts[3] = VertexNT(offset+vec3(0,info.bh,0),
                            vec3(0,0,1),
                            vec2(info.tcMin.x,info.tcMin.y));

        x+=info.ax;
        y+=info.ay;
        mesh.addQuad(verts);
    }
}

DynamicText* TextGenerator::createDynamicText(int size, bool normalize){
    DynamicText* text = new DynamicText(size);

    text->texture = textureAtlas;

    std::string buffer;
    buffer.resize(size);
    buffer.assign(size,'A');

    createTextMesh(text->mesh,buffer);

    if(normalize){
        text->mesh.boundingBox.growBox(maxCharacter);
        aabb bb = text->mesh.getAabb();
        vec3 offset = bb.getPosition();
        mat4 t;
        t[3] = vec4(-offset,0);
        text->mesh.transform(t);
    }
    text->mesh.createBuffers(text->buffer);


    text->label = buffer;

    return text;
}

Text* TextGenerator::createText(const std::string &label, bool normalize){
    Text* text = new Text(label);

    text->texture = textureAtlas;

    createTextMesh(text->mesh,label);

    if(normalize){
        text->mesh.boundingBox.growBox(maxCharacter);
        aabb bb = text->mesh.getAabb();
        vec3 offset = bb.getPosition();
        mat4 t;
        t[3] = vec4(-offset,0);
        text->mesh.transform(t);
    }
    text->mesh.createBuffers(text->buffer);

    return text;
}

void TextGenerator::updateText(DynamicText* text, const std::string &l, int startIndex){
    std::string label(l);
    text->compressText(label,startIndex);
    if(label.size()==0){
        //no update needed
        return;
    }


    character_info &info = characters[(int)text->label[startIndex]];
    text->updateText(label,startIndex);

    //x offset of first new character
    int start = text->mesh.vertices[startIndex*4].position.x - info.bl;
    //delete everything from startindex to end
    text->mesh.vertices.resize(startIndex*4);
    text->mesh.faces.resize(startIndex);


    //calculate new faces
    createTextMesh(text->mesh,label,start);

    //update gl mesh
    text->updateGLBuffer(startIndex);


}
