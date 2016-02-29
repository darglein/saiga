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
class Text;

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
public:
    static FT_Library ft;
    FT_Face face = nullptr;

    //distance between characters in texture atlas
    int charPaddingX = 5;
    int charPaddingY = 5;

    //additional border pixels (usefull for border lines)
    int charBorder = 5;

    struct character_info {
      int ax = 0; // advance.x
      int ay = 0; // advance.y

      int bw = 0; // bitmap.width;
      int bh = 0; // bitmap.rows;

      int bl = 0; // bitmap_left;
      int bt = 0; // bitmap_top;

      int atlasX = 0, atlasY = 0; //position of this character in the texture atlas
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

//    Text* createDynamicText(int size, bool normalize=false);
//    //if normalized-> origin of the textmesh is in the center of the text
//    Text* createText(const std::string &label, bool normalize=false);
};
