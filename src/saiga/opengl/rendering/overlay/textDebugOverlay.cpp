/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"
#ifdef SAIGA_USE_FREETYPE
#    include "saiga/core/geometry/triangle_mesh.h"
#    include "saiga/opengl/framebuffer.h"
#    include "saiga/opengl/rendering/overlay/textDebugOverlay.h"
#    include "saiga/opengl/shader/basic_shaders.h"
#    include "saiga/opengl/shader/shaderLoader.h"
#    include "saiga/opengl/text/text.h"
#    include "saiga/opengl/text/textAtlas.h"
#    include "saiga/opengl/text/textShader.h"

namespace Saiga
{
TextDebugOverlay::TextDebugOverlay(int w, int h) : overlay(1, 1), layout(w, h) {}

TextDebugOverlay::~TextDebugOverlay()
{
    for (TDOEntry& entry : entries)
    {
        delete entry.text;
    }
}

void TextDebugOverlay::init(TextAtlas* textureAtlas)
{
    this->textureAtlas = textureAtlas;
}

void TextDebugOverlay::render()
{
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    overlay.render();
}

int TextDebugOverlay::createItem(const std::string& name)
{
    int id = entries.size();
    TDOEntry entry;
    entry.valueIndex = name.size();

    entry.text         = new Text(textureAtlas, "");
    entry.text->params = textParameters;
    overlay.addText(entry.text);

    entry.text->updateText(name, 0);
    AABB bb = entry.text->getAabb();
    bb.growBox(textureAtlas->getMaxCharacter());


    int y = id;

    vec2 relPos(0, 0);
    relPos[0] = borderX;
    //    relPos[0] = 0.5;
    relPos[1] = 1.0f - ((y) * (paddingY + textSize) + borderY);

    layout.transform(entry.text, bb, relPos, textSize, Layout::LEFT, Layout::RIGHT);

    entries.push_back(entry);


    entry.text->updateText("123", entry.valueIndex);

    return id;
}


}  // namespace Saiga
#endif
