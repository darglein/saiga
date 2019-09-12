/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/text/TextOverlay2D.h"

#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/opengl/text/text.h"
#include "saiga/opengl/text/textShader.h"

#include <algorithm>

namespace Saiga
{
TextOverlay2D::TextOverlay2D(const mat4& proj)
{
    this->proj = proj;
    loadShader();
}

TextOverlay2D::TextOverlay2D(int width, int height) : width(width), height(height)
{
    this->proj = ortho(0.0f, (float)width, 0.0f, (float)height, -1.0f, 1.0f);
    loadShader();
}

void TextOverlay2D::render()
{
    textShader->bind();
    for (Text*& text : texts)
    {
        if (text->visible) text->render(textShader);
    }
    textShader->unbind();
}

void TextOverlay2D::render(Camera* camera)
{
    textShader->bind();
    for (Text*& text : texts)
    {
        if (text->visible) text->render(textShader);
    }
    textShader->unbind();
}

void TextOverlay2D::addText(Text* text)
{
    texts.push_back(text);
}

void TextOverlay2D::removeText(Text* text)
{
    texts.erase(std::remove(texts.begin(), texts.end(), text), texts.end());
}

void TextOverlay2D::loadShader()
{
    if (textShader != nullptr) return;
    textShader = shaderLoader.load<TextShader>("sdf_text.glsl");
}

}  // namespace Saiga
