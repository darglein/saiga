/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <saiga/config.h>
#include "saiga/rendering/program.h"
#include "saiga/opengl/uniformBuffer.h"
#include "saiga/imgui/imgui_renderer.h"

namespace Saiga {


class SAIGA_GLOBAL Renderer{
public:
    std::shared_ptr<ImGuiRenderer> imgui;
    Rendering* rendering = nullptr;
    int outputWidth = -1, outputHeight = -1;
    UniformBuffer cameraBuffer;

    Renderer(OpenGLWindow &window);
    virtual ~Renderer();

    virtual void renderImGui(bool* p_open = NULL) {}
    virtual float getTotalRenderTime() {return 0;}

    virtual void resize(int windowWidth, int windowHeight);
    virtual void render_intern(Camera *cam) = 0;

    void setRenderObject(Rendering &r );
    virtual void printTimings() {}

    void bindCamera(Camera* cam);
};

}
