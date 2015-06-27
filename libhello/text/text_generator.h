#pragma once
#include "libhello/opengl/vertexBuffer.h"
#include "libhello/geometry/triangle_mesh.h"
//#include "libhello/opengl/mesh.h"
#include "libhello/text/text.h"
#include "libhello/text/dynamic_text.h"


#include <iostream>

//forward declarations to avoid including the ft header
//with that the ft library only has to be linked to the framework
struct FT_FaceRec_;
struct FT_LibraryRec_;
typedef struct FT_FaceRec_*  FT_Face;
typedef struct FT_LibraryRec_  *FT_Library;


class TextGenerator{
private:
    static FT_Library ft;
    FT_Face face = nullptr;
    int charOffset = 5; //distance between characters in texture atlas

    struct character_info {
      int ax; // advance.x
      int ay; // advance.y

      int bw; // bitmap.width;
      int bh; // bitmap.rows;

      int bl; // bitmap_left;
      int bt; // bitmap_top;

      int atlasX, atlasY; //position of this character in the texture atlas
      vec2 tcMin,tcMax;
    } characters[128];

    void createTextureAtlas();

     basic_Texture_2D *textureAtlas = nullptr;
public:
    aabb maxCharacter;
    std::string font;
    int font_size;
    TextGenerator();
    ~TextGenerator();
    void loadFont(const std::string &font, int font_size);

    void createTextMesh(TriangleMesh<VertexNT, GLuint> &mesh, const std::string &text, int startX=0, int startY=0);
    DynamicText* createDynamicText(int size, bool normalize=false);

    //if normalized-> origin of the textmesh is in the center of the text
    Text* createText(const std::string &label, bool normalize=false);
    void updateText(DynamicText* text, const std::string &label, int startIndex);
};
