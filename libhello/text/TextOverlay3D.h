#pragma once

#include "libhello/util/glm.h"

#include <vector>
#include <memory>

class TextShader;
class Text;
class Camera;

class TextOverlay3D {

public:

    struct TextContainer{
        std::unique_ptr<Text> text;
        float duration = 0.f;
        bool orientToCamera = true;

        TextContainer(std::unique_ptr<Text> text, float duration, bool orientToCamera)
            : text(std::move(text)), duration(duration), orientToCamera(orientToCamera){}

        TextContainer(TextContainer &&other) : text(std::move(other.text)), duration(other.duration), orientToCamera(other.orientToCamera)
        {
        }

        TextContainer& operator=(TextContainer &&other)
        {
          text = std::move(other.text);
          this->duration = other.duration;
          this->orientToCamera = other.orientToCamera;
          return *this;
        }

    };

    TextShader* textShader;
    //text + duration
    std::vector<TextContainer> texts;

    static const float INFINITE_DURATION;


    TextOverlay3D();
    void render(Camera* cam);
    void renderText(Camera *cam);

    //text stuff
    void addText(std::unique_ptr<Text> text, float duration, bool orientToCamera = true);

    void update(float secondsPerTick);


    void setTextShader(TextShader* textShader);
};


