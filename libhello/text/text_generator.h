#pragma once
#include "libhello/opengl/vertexBuffer.h"
#include "libhello/geometry/triangle_mesh.h"
#include "libhello/opengl/mesh.h"
#include "libhello/text/text.h"
#include "libhello/text/dynamic_text.h"

#include <ft2build.h>
#include FT_FREETYPE_H
#include <iostream>


class TextGenerator{
private:
    static FT_Library* ft;
    FT_Face face;
    int charOffset = 5; //distance between characters in texture atlas

    struct character_info {
      int ax; // advance.x
      int ay; // advance.y

      int bw; // bitmap.width;
      int bh; // bitmap.rows;

      int bl; // bitmap_left;
      int bt; // bitmap_top;

      vec2 tcMin,tcMax;
    } characters[128];

    void createTextureAtlas();

     Texture *textureAtlas = nullptr;
public:

    string font;
    int font_size;
    TextGenerator();
    ~TextGenerator();
    void loadFont(const string &font, int font_size);

    void createTextMesh(TriangleMesh<VertexNT, GLuint> &mesh, const string &text, int startX=0, int startY=0);
    DynamicText* createDynamicText(int size);
    Text* createText(const string &label);
    void updateText(DynamicText* text, const string &label, int startIndex);
};
