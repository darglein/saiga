#pragma once

#include "libhello/util/glm.h"

#include <vector>
#include <memory>

class TextShader;
class Text;
class Camera;

class TextOverlay3D {
public:

    TextShader* textShader;
    //text + duration
    std::vector<std::pair<std::unique_ptr<Text>, float>> texts;



    TextOverlay3D();
    void render(Camera* cam);
    void renderText(Camera *cam);

    //text stuff
    void addText(std::unique_ptr<Text> text, float duration);

    void update(float secondsPerTick);


    void setTextShader(TextShader* textShader);
};


