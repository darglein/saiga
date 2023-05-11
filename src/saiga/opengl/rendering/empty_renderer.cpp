/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "empty_renderer.h"

#include "saiga/core/camera/camera.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/opengl/assets/asset.h"
#include "saiga/opengl/framebuffer.h"
#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/opengl/window/OpenGLWindow.h"


namespace Saiga
{
EmptyRenderer::EmptyRenderer(OpenGLWindow& window, const ParameterType& params)
    : OpenGLRenderer(window, params.createRendererMainMenuItem, params.createLogMainMenuItem), params(params)
{
    render_viewport = false;
    std::cout << " EmptyRenderer initialized. Render resolution: " << window.getWidth() << "x" << window.getHeight()
              << std::endl;
}

void EmptyRenderer::renderGL(Framebuffer* target_framebuffer, ViewPort viewport, Camera* camera)

{
    if (!rendering) return;

    assert_no_glerror();
}

void EmptyRenderer::Resize(int windowWidth, int windowHeight)
{
    if (windowWidth == renderWidth && windowHeight == renderHeight)
    {
        // Already at correct size
        // -> Skip resize
        return;
    }

    this->renderWidth  = windowWidth;
    this->renderHeight = windowHeight;
    std::cout << "Resizing Window to : " << windowWidth << "," << windowHeight << std::endl;
    std::cout << "Framebuffer size: " << renderWidth << " " << renderHeight << std::endl;

}

void EmptyRenderer::renderImgui()
{
    if (!should_render_imgui) return;

}


}  // namespace Saiga
