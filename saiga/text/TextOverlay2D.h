#pragma once

#include "saiga/config.h"
#include "saiga/util/glm.h"
#include "saiga/camera/camera.h"

#include <vector>

class TextShader;
class Text;

class SAIGA_GLOBAL TextOverlay2D {
public:

    mat4 proj;
    std::vector<Text*> texts;

    int width,height;

    TextOverlay2D();
    TextOverlay2D(int width=1, int height=1);
    void render();
    void render(Camera* camera);

    //text stuff
    void addText(Text* text);
    void removeText(Text* text);

    void loadShader();


private:
    TextShader* textShader = nullptr;
};


