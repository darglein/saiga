#pragma once

#include "libhello/util/glm.h"

#include <vector>

class TextShader;
class Text;

class TextOverlay2D {
public:

    mat4 proj;
    TextShader* textShader;
    std::vector<Text*> texts;

    int width,height;

    TextOverlay2D(int width, int height);
    void render();

    //text stuff
    void addText(Text* text);
    void setTextShader(TextShader* textShader);
};


