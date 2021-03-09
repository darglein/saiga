/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/window/Interfaces.h"
#include "saiga/opengl/framebuffer.h"
#include "saiga/opengl/imgui/imgui_opengl.h"
#include "saiga/opengl/rendering/program.h"
#include "saiga/opengl/uniformBuffer.h"

namespace Saiga
{
struct SAIGA_OPENGL_API RenderingParameters
{
    // adds a 'glfinish' at the end of the rendering. usefull for debugging.
    bool useGlFinish = true;


    vec4 clearColor = vec4(0, 0, 0, 0);

    bool wireframe          = false;
    float wireframeLineSize = 1;

    /**
     * Setting this to true means the user takes care of viewport updates during window resize.
     * Currently only makes sense for multi view rendering (split screen)
     */
    bool userViewPort = false;

    /**
     *  Reads all paramters from the given config file.
     *  Creates the file with the default values if it doesn't exist.
     */
    void fromConfigFile(const std::string& file);
};


class SAIGA_OPENGL_API OpenGLRenderer : public RendererBase
{
   public:
    OpenGLRenderer(OpenGLWindow& window);
    virtual ~OpenGLRenderer();


    void render(const RenderInfo& renderInfo) override;

    virtual void renderGL(Framebuffer* target_framebuffer, ViewPort viewport, Camera* camera) {}


    void ResizeTarget(int windowWidth, int windowHeight);
    virtual void printTimings() override {}
    void bindCamera(Camera* cam);

    std::shared_ptr<ImGui_GL_Renderer> imgui;

    int outputWidth = -1, outputHeight = -1;
    UniformBuffer cameraBuffer;
    OpenGLWindow* window;

    // The target framebuffer, where all derived renders will render into
    // This buffer is passed to the renderGL() function
    std::unique_ptr<Framebuffer> target_framebuffer;
    Framebuffer default_framebuffer;

    // A bool that is true if the 3D viewport is in focus or an imgui element
    // This should be used by the application to filter keyborad/mouse input
    bool is_viewport_focused = true;
};

inline void setViewPort(const ViewPort& vp)
{
    glViewport(vp.position(0), vp.position(1), vp.size(0), vp.size(1));
}

inline void setScissor(const ViewPort& vp)
{
    glScissor(vp.position(0), vp.position(1), vp.size(0), vp.size(1));
}


inline ViewPort getViewPort()
{
    GLint gl_vp[4];
    glGetIntegerv(GL_VIEWPORT, gl_vp);

    ViewPort result;
    result.position(0) = gl_vp[0];
    result.position(1) = gl_vp[1];
    result.size(0)     = gl_vp[2];
    result.size(1)     = gl_vp[3];
    return result;
}

}  // namespace Saiga
