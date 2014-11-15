#pragma once

#include "libhello/util/glm.h"
#include "libhello/opengl/shader.h"
#include "libhello/text/text.h"
#include <vector>

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


