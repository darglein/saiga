#pragma once

#include "saiga/config.h"
#include "saiga/util/glm.h"

#include <vector>

class TextShader;
class Text;

class SAIGA_GLOBAL TextOverlay2D {
public:

    mat4 proj;

    std::vector<Text*> texts;

    int width,height;

    TextOverlay2D(const mat4& proj);
    TextOverlay2D(int width=1, int height=1);
    void render();

    //text stuff
    void addText(Text* text);
    void removeText(Text* text);

    void loadShader();


private:
    TextShader* textShader = nullptr;
};


