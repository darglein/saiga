#pragma once

#include "libhello/util/glm.h"

#include <vector>

class TextShader;
class Text;

class Overlay {
public:

    mat4 proj;
    TextShader* textShader;
    std::vector<Text*> texts;

    int width,height;
    Overlay(int width, int height);
    void render();

    //text stuff
    void addText(Text* text);
    void renderText();
    void setTextShader(TextShader* textShader);
};


