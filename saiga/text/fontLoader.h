#pragma once

//#include "saiga/geometry/triangle_mesh.h"
//#include "saiga/text/text.h"
//#include "saiga/text/dynamic_text.h"

#include "saiga/config.h"
#include "saiga/util/glm.h"
#include "saiga/geometry/aabb.h"
#include "saiga/opengl/vertex.h"
#include "saiga/opengl/opengl.h"
#include "saiga/opengl/texture/image.h"

#include <iostream>

struct FT_FaceRec_;
struct FT_LibraryRec_;
typedef struct FT_FaceRec_*  FT_Face;
typedef struct FT_LibraryRec_  *FT_Library;


class FontLoader{
public:
    struct Glyph{
        int character; //in ascii encoding
        glm::vec2 advance; //distance to the origin of the next character
        glm::vec2 offset;  //offset of the bitmap position to the origin of this character
        glm::vec2 size; //size of bitmap
        Image* bitmap;
    };

    std::vector<Glyph> glyphs;

    FontLoader(const std::string& file);
    void loadMonochromatic(int fontSize, int glyphPadding = 0);
    void writeGlyphsToFiles(const std::string& prefix);

private:
    std::string file;
    FT_Face face = nullptr;

    void loadFace(int fontSize);
    void addGlyph(int gindex, int glyphPadding);
};
