/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/imgui/imgui_renderer.h"
#include "saiga/opengl/uniformBuffer.h"
#include "saiga/rendering/program.h"
#include "saiga/window/Interfaces.h"

namespace Saiga
{
struct SAIGA_GLOBAL RenderingParameters
{
    /**
     * If srgbWrites is enabled all writes to srgb textures will cause a linear->srgb converesion.
     * Important to note is that writes to the default framebuffer also be converted to srgb.
     * This means if srgbWrites is enabled all shader inputs must be converted to linear rgb.
     * For textures use the srgb flag.
     * For vertex colors and uniforms this conversion must be done manually with Color::srgb2linearrgb()
     *
     * If srgbWrites is disabled the gbuffer and postprocessor are not allowed to have srgb textures.
     *
     * Note: If srgbWrites is enabled, you can still use a non-srgb gbuffer and post processor.
     */
    bool srgbWrites = true;


    // adds a 'glfinish' at the end of the rendering. usefull for debugging.
    bool useGlFinish = false;


    vec4 clearColor = vec4(0, 0, 0, 0);

    bool wireframe          = false;
    float wireframeLineSize = 1;
};


class SAIGA_GLOBAL Renderer : public RendererBase
{
   public:
    std::shared_ptr<ImGuiRenderer> imgui;

    int outputWidth = -1, outputHeight = -1;
    UniformBuffer cameraBuffer;

    Renderer(OpenGLWindow& window);
    virtual ~Renderer();


    virtual void resize(int windowWidth, int windowHeight);


    virtual void printTimings() {}

    void bindCamera(Camera* cam);
};

}  // namespace Saiga
