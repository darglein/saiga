#pragma once

#include "libhello/util/glm.h"

#include <vector>

class TextShader;
class Text;
class Camera;

class TextOverlay3D {
public:

    TextShader* textShader;
    //text + duration
    std::vector<std::pair<Text*, float>> texts;



    TextOverlay3D();
    void render(Camera* cam);
    void renderText(Camera *cam);

    //text stuff
    void addText(Text* text, float duration);

    void update(float secondsPerTick);


    void setTextShader(TextShader* textShader);
};


