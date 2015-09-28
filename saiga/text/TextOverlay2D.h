#pragma once

#include "saiga/config.h"
#include "saiga/util/glm.h"

#include <vector>

class TextShader;
class Text;

class SAIGA_GLOBAL TextOverlay2D {
public:

    mat4 proj;
    TextShader* textShader;
    std::vector<Text*> texts;

    int width,height;

    TextOverlay2D(int width=1, int height=1);
    void render();

    //text stuff
    void addText(Text* text);
    void removeText(Text* text);
    void setTextShader(TextShader* textShader);
};


