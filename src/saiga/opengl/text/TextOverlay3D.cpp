/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/text/TextOverlay3D.h"

#include "saiga/core/camera/camera.h"
#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/opengl/text/text.h"
#include "saiga/opengl/text/textShader.h"

#include <algorithm>

namespace Saiga
{
const float TextOverlay3D::INFINITE_DURATION = -1.f;

bool TextOverlay3D::TextContainer::update(float delta)
{
    //    text->position += vec4(velocity*delta,0);
    // text->translateGlobal(velocity*delta);

    //    text->position += vec4(vec3(velocity*delta),0);



    if (duration == INFINITE_DURATION) return false;
    duration -= delta;
    return duration <= 0;
}

void TextOverlay3D::TextContainer::interpolate(float interpolationInSeconds, const mat4& v)
{
    float timeAlive = (maxDuration - duration) + interpolationInSeconds;
    if (timeAlive < timeToUpscale)
    {
        float factor = (timeAlive) / timeToUpscale;
        text->scale  = vec4(startScale * (1 - factor) + factor * targetScale);
    }
    else
    {
        text->scale = vec4(targetScale);
    }

    text->position = startPosition + make_vec4(vec3(timeAlive * velocity), 0);

    text->params.setAlpha(getFade(interpolationInSeconds));

    text->calculateModel();
    if (orientToCamera)
    {
        // make this text face towards the camera
        text->model = text->model * v;
    }
}

float TextOverlay3D::TextContainer::getFade(float interpolationInSeconds)
{
    float a = 1.0f;
    if (fade)
    {
        if (duration < fadeStart) a = (duration + interpolationInSeconds) / (maxDuration - fadeStart);
    }
    return a;
}


TextOverlay3D::TextOverlay3D() {}

void TextOverlay3D::render(Camera* cam, float interpolationInSeconds)
{
    renderText(cam, interpolationInSeconds);
}

void TextOverlay3D::addText(std::shared_ptr<Text> text, float duration, bool orientToCamera)
{
    texts.push_back(TextContainer(std::move(text), duration, orientToCamera));
}

void TextOverlay3D::addText(const TextOverlay3D::TextContainer& tc)
{
    texts.push_back(tc);
}

void TextOverlay3D::update(float secondsPerTick)
{
    auto func = [secondsPerTick](TextContainer& p) { return p.update(secondsPerTick); };

    texts.erase(std::remove_if(texts.begin(), texts.end(), func), texts.end());
}



void TextOverlay3D::renderText(Camera* cam, float interpolationInSeconds)
{
    if(textShader->bind())
    {
        mat4 v   = cam->model;
        v.col(3) = vec4(0, 0, 0, 1);


        for (TextContainer& p : texts)
        {
            //        textShader->uploadFade(p.getFade());
            p.interpolate(interpolationInSeconds, v);



            p.text->render(textShader);
        }
        textShader->unbind();
    }
}


void TextOverlay3D::loadShader()
{
    if (textShader != nullptr) return;
    //    textShader = shaderLoader.load<TextShaderFade>("deferred_text3D.glsl");
    textShader = shaderLoader.load<TextShader>("sdf_text.glsl");
}

}  // namespace Saiga
