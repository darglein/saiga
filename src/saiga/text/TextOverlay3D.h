/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/util/math.h"

#include <vector>

namespace Saiga
{
class TextShader;
class Text;
class Camera;
class TextShaderFade;

class SAIGA_GLOBAL TextOverlay3D
{
   public:
    struct TextContainer
    {
        std::shared_ptr<Text> text;
        float fadeStart     = 0.0f;
        float duration      = 0.f;
        float maxDuration   = 0.0f;
        bool orientToCamera = true;
        vec3 velocity       = vec3(0);
        bool fade           = false;
        vec4 startPosition  = vec4(0);

        float timeToUpscale = 1.0f;
        vec4 targetScale    = vec4(1.f);
        vec4 startScale     = vec4(1.f);
        vec4 upscale        = vec4(0);

        TextContainer() {}
        TextContainer(std::shared_ptr<Text> text, float duration, bool orientToCamera)
            : text(text), duration(duration), maxDuration(duration), orientToCamera(orientToCamera)
        {
        }

        bool update(float delta);
        void interpolate(float interpolation, const mat4& v);
        float getFade(float interpolationInSeconds);
    };


    // text + duration
    std::vector<TextContainer> texts;

    static const float INFINITE_DURATION;


    TextOverlay3D();
    void render(Camera* cam, float interpolation);
    void renderText(Camera* cam, float interpolationInSeconds);

    // text stuff
    void addText(std::shared_ptr<Text> text, float duration, bool orientToCamera = true);
    void addText(const TextContainer& tc);


    void update(float secondsPerTick);

    void loadShader();

   private:
    std::shared_ptr<TextShader> textShader = nullptr;
};

}  // namespace Saiga
