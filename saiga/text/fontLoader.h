#pragma once

#include "saiga/config.h"
#include "saiga/util/glm.h"
#include "saiga/opengl/texture/image.h"

#include <iostream>
#include <vector>

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
        Image* bitmap = nullptr;
    };

    std::vector<Glyph> glyphs;

    FontLoader(const std::string& file);
    ~FontLoader();
    void loadMonochromatic(int fontSize, int glyphPadding = 0);
    void writeGlyphsToFiles(const std::string& prefix);

private:
    static FT_Library ft;
    std::string file;
    FT_Face face = nullptr;

    void loadFace(int fontSize);
    void addGlyph(int gindex, int glyphPadding);
};
