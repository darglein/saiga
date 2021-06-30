/**
 * Copyright (c) 2021 Darius Rückert
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
TextOverlay2D::TextOverlay2D(int width, int height) : layout(width, height)
{
    loadShader();
}

void TextOverlay2D::render()
{
    if(textShader->bind())
    {
        for (Text*& text : texts)
        {
            if (text->visible) text->render(textShader);
        }
        textShader->unbind();
    }
}


void TextOverlay2D::addText(Text* text)
{
    texts.push_back(text);
}

void TextOverlay2D::removeText(Text* text)
{
    texts.erase(std::remove(texts.begin(), texts.end(), text), texts.end());
}

void TextOverlay2D::PositionText2d(Text* text, vec2 position, float size, Layout::Alignment alignmentX,
                                   Layout::Alignment alignmentY)
{
    AABB bb = text->getAabb();
    layout.transform(text, bb, position, size, alignmentX, alignmentY);
}

void TextOverlay2D::loadShader()
{
    if (textShader != nullptr) return;
    textShader = shaderLoader.load<TextShader>("sdf_text.glsl");
}

}  // namespace Saiga
