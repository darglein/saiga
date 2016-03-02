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
class Image;
//forward declarations to avoid including the ft header
//with that the ft library only has to be linked to the framework
struct FT_FaceRec_;
struct FT_LibraryRec_;
typedef struct FT_FaceRec_*  FT_Face;
typedef struct FT_LibraryRec_  *FT_Library;


class SAIGA_GLOBAL TextureAtlas{
public:
    struct character_info {
        int ax = 0; // advance.x
        int ay = 0; // advance.y

        int bw = 0; // bitmap.width;
        int bh = 0; // bitmap.rows;

        int bl = 0; // bitmap_left;
        int bt = 0; // bitmap_top;

        int atlasX = 0, atlasY = 0; //position of this character in the texture atlas
        vec2 tcMin,tcMax;
    } ;

    TextureAtlas();
    ~TextureAtlas();

    /**
     * Loads a True Type font (.ttf) with libfreetype.
     * This will create the textureAtlas, so it has to be called before any ussage.
     */
    void loadFont(const std::string &font, int font_size, int stroke_size=0);

    /**
     * Returns the bounding box that could contain every character in this font.
     */
    aabb getMaxCharacter(){return maxCharacter;}

    /**
     * Returns the actual opengl texture.
     */
    basic_Texture_2D *getTexture(){return textureAtlas;}

    /**
     * Returns information to a specific character in this font.
     */
    const character_info& getCharacterInfo(int c){ return characters[c];}

    static FT_Library ft;
private:

    FT_Face face = nullptr;

    //distance between characters in texture atlas
    int charPaddingX = 5;
    int charPaddingY = 5;

    //additional border pixels (usefull for border lines)
    //    int charBorder = 5;
    int atlasHeight;
    int atlasWidth;
    int charNum;

    character_info characters[128];

    basic_Texture_2D *textureAtlas = nullptr;
    aabb maxCharacter;
    std::string font;
    int font_size;
    int stroke_size;


    void createTextureAtlas(Image &outImg);

    //mono chromatic image without strokes
    void createTextureAtlasMono(Image &outImg);

    void createTextureAtlasSDF(Image &moneImage, Image &outImg);
    //calculates the size and the position of each individual character in the texture atlas.
    void calculateTextureAtlasPositions();
};
