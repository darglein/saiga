/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/camera/camera.h"
#include "saiga/core/math/math.h"
#include "saiga/opengl/rendering/overlay/Layout.h"

#include <memory>
#include <vector>

namespace Saiga
{
class TextShader;
class Text;

class SAIGA_OPENGL_API TextOverlay2D
{
   public:
    std::vector<Text*> texts;

    TextOverlay2D(int width = 1, int height = 1);
    void render();

    // text stuff
    void addText(Text* text);
    void removeText(Text* text);

    Camera* GetCamera() { return &layout.cam; }

    void PositionText2d(Text* text, vec2 position, float size, Layout::Alignment alignmentX = Layout::LEFT,
                        Layout::Alignment alignmentY = Layout::LEFT);

   private:
    std::shared_ptr<TextShader> textShader = nullptr;
    Layout layout;
    void loadShader();
};

}  // namespace Saiga
