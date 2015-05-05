#pragma once

#include "libhello/util/glm.h"

#include <vector>

class TextShader;
class Text;
class Camera;

class TextOverlay3D {
public:

    TextShader* textShader;
    std::vector<Text*> texts;



    TextOverlay3D();
    void render(Camera* cam);
    void renderText(Camera *cam);

    //text stuff
    void addText(Text* text);

    void setTextShader(TextShader* textShader);
};


