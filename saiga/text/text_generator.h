#pragma once

//#include "saiga/geometry/triangle_mesh.h"
//#include "saiga/text/text.h"
//#include "saiga/text/dynamic_text.h"

#include "saiga/config.h"
#include "saiga/util/glm.h"
#include "saiga/geometry/aabb.h"
#include "saiga/opengl/vertex.h"
#include "saiga/opengl/opengl.h"

#include <iostream>


class Text;
class DynamicText;

template<typename vertex_t, typename index_t>
class TriangleMesh;

class basic_Texture_2D;

//forward declarations to avoid including the ft header
//with that the ft library only has to be linked to the framework
struct FT_FaceRec_;
struct FT_LibraryRec_;
typedef struct FT_FaceRec_*  FT_Face;
typedef struct FT_LibraryRec_  *FT_Library;


class SAIGA_GLOBAL TextGenerator{
private:
    static FT_Library ft;
    FT_Face face = nullptr;

    //distance between characters in texture atlas
    int charPaddingX = 5;
    int charPaddingY = 5;

    //additional border pixels (usefull for border lines)
    int charBorder = 5;

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
    int stroke_size;

    TextGenerator();
    ~TextGenerator();

    void loadFont(const std::string &font, int font_size, int stroke_size=0);

    void createTextMesh(TriangleMesh<VertexNT, GLuint> &mesh, const std::string &text, int startX=0, int startY=0);
    DynamicText* createDynamicText(int size, bool normalize=false);

    //if normalized-> origin of the textmesh is in the center of the text
    Text* createText(const std::string &label, bool normalize=false);
    void updateText(DynamicText* text, const std::string &label, int startIndex);
};
