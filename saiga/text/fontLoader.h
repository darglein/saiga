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

#include <ft2build.h>
#include <ftstroke.h>


class FontLoader{
public:
    struct Glyph{
        int character; //in ascii encoding
        glm::vec2 advance; //distance to the origin of the next character
        glm::vec2 offset;  //offset of the bitmap position to the origin of this character
        Image* bitmap;
    };

    std::vector<Glyph> glyphs;

    FontLoader(const std::string& file);
    void loadMonochromatic(int fontSize);
    void writeGlyphsToFiles(const std::string& prefix);

     void loadMonochromatic2(int fontSize);
private:
    std::string file;
    FT_Face face = nullptr;

    void loadFace(int fontSize);
    void addGlyph(int gindex);
};
