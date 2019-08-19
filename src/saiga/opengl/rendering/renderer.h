/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/window/Interfaces.h"
#include "saiga/opengl/imgui/imgui_opengl.h"
#include "saiga/opengl/rendering/program.h"
#include "saiga/opengl/uniformBuffer.h"

namespace Saiga
{
struct SAIGA_OPENGL_API RenderingParameters
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

    /**
     * Setting this to true means the user takes care of viewport updates during window resize.
     * Currently only makes sense for multi view rendering (split screen)
     */
    bool userViewPort = false;
};


class SAIGA_OPENGL_API OpenGLRenderer : public RendererBase
{
   public:
    std::shared_ptr<ImGui_GL_Renderer> imgui;

    int outputWidth = -1, outputHeight = -1;
    UniformBuffer cameraBuffer;

    OpenGLRenderer(OpenGLWindow& window);
    virtual ~OpenGLRenderer();


    virtual void resize(int windowWidth, int windowHeight);


    virtual void printTimings() {}

    void bindCamera(Camera* cam);
};

inline void setViewPort(const ViewPort& vp)
{
    glViewport(vp.position(0), vp.position(1), vp.size(0), vp.size(1));
}

}  // namespace Saiga
