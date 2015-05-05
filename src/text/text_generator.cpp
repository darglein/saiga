#include "text/text_generator.h"
#include <algorithm>
#define NOMINMAX
#undef max

FT_Library* TextGenerator::ft = NULL;

TextGenerator::TextGenerator(){

    if(ft==NULL){
        ft = new FT_Library();
        if(FT_Init_FreeType(ft)) {
            cerr<< "Could not init freetype library"<<endl;
            exit(1);
        }
    }
}

TextGenerator::~TextGenerator()
{
    delete textureAtlas;
}


void TextGenerator::loadFont(const string &font, int font_size){
    this->font = font;
    this->font_size = font_size;

    if(FT_New_Face(*ft, font.c_str(), 0, &face)) {
        cerr<<"Could not open font\n"<<endl;
        return;
    }
    FT_Set_Pixel_Sizes(face, 0, font_size);

    createTextureAtlas();
}

void TextGenerator::createTextureAtlas(){
    int chars= 0;
    int w=0,h=0;
    FT_GlyphSlot g = face->glyph;
    for(int i = 32; i < 128; i++) {
        if(FT_Load_Char(face, i, FT_LOAD_RENDER)) {
            cerr<<"Could not load character '"<<(char)i<<"'"<<endl;
            continue;
        }

        w += g->bitmap.width+charOffset;
        h = std::max(h, (int)g->bitmap.rows);
        chars++;
    }


    std::vector<unsigned char> data(w*h,0);

    textureAtlas = new Texture();

    //zero initialize texture
    textureAtlas->createTexture(w ,h,GL_RED, GL_R8  ,GL_UNSIGNED_BYTE,&data[0]);


    textureAtlas->bind();
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    textureAtlas->unbind();
    int x = 0;

    for(int i = 32; i < 128; i++) {
        if(FT_Load_Char(face, i, FT_LOAD_RENDER))
            continue;
        character_info &info = characters[i];

        info.ax = g->advance.x >> 6;
        info.ay = g->advance.y >> 6;

        info.bw = g->bitmap.width;
        info.bh = g->bitmap.rows;

        info.bl = g->bitmap_left;
        info.bt = g->bitmap_top;


        float tx = (float)x / (float)w;
        info.tcMin = vec2(tx,0);
        info.tcMax = vec2(tx+(float)info.bw/(float)textureAtlas->getWidth(),(float)info.bh/(float)textureAtlas->getHeight());

        textureAtlas->uploadSubImage(x, 0, g->bitmap.width, g->bitmap.rows, g->bitmap.buffer);
        x += g->bitmap.width+charOffset;

    }
}



void TextGenerator::createTextMesh(TriangleMesh<VertexNT, GLuint> &mesh, const string &text, int startX, int startY){

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

DynamicText* TextGenerator::createDynamicText(int size){
    DynamicText* text = new DynamicText(size);

    text->texture = textureAtlas;

    string buffer;
    buffer.resize(size);
    buffer.assign(size,'A');

    createTextMesh(text->mesh,buffer);
    text->mesh.createBuffers(text->buffer);
    text->label = buffer;

    return text;
}

Text* TextGenerator::createText(const string &label, bool normalize){
    Text* text = new Text(label);

    text->texture = textureAtlas;

    createTextMesh(text->mesh,label);

    if(normalize){
        aabb bb = text->mesh.getAabb();
        vec3 offset = bb.getPosition();
        mat4 t;
        t[3] = vec4(-offset,0);
        text->mesh.transform(t);
    }


    text->mesh.createBuffers(text->buffer);

    return text;
}

void TextGenerator::updateText(DynamicText* text, const string &l, int startIndex){
    string label(l);
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
