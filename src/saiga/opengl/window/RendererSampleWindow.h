/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/glfw/all.h"
#include "saiga/opengl/assets/all.h"
#include "saiga/opengl/rendering/deferredRendering/deferredRendering.h"
#include "saiga/opengl/rendering/deferredRendering/uberDeferredRendering.h"
#include "saiga/opengl/rendering/forwardRendering/forwardRendering.h"
#include "saiga/opengl/rendering/renderer.h"
#include "saiga/opengl/window/WindowTemplate.h"
#include "saiga/opengl/window/glfw_window.h"
#include "saiga/opengl/world/proceduralSkybox.h"



namespace Saiga
{
/**
 * This is the class from which saiga samples can inherit from.
 * It's a basic scene with a camera and a skybox.
 *
 * @brief The RendererSampleWindow class
 */

//#define SINGLE_PASS_DEFERRED_PIPELINE
//#define MULTI_PASS_DEFERRED_PIPELINE
#define SINGLE_PASS_FORWARD_PIPELINE

#ifdef SINGLE_PASS_DEFERRED_PIPELINE
#    undef MULTI_PASS_DEFERRED_PIPELINE
#    undef SINGLE_PASS_FORWARD_PIPELINE
class SAIGA_OPENGL_API RendererSampleWindow : public StandaloneWindow<WindowManagement::GLFW, UberDeferredRenderer>,
                                              public glfw_KeyListener
#elif defined(MULTI_PASS_DEFERRED_PIPELINE)
#    undef SINGLE_PASS_FORWARD_PIPELINE
class SAIGA_OPENGL_API RendererSampleWindow : public StandaloneWindow<WindowManagement::GLFW, DeferredRenderer>,
                                              public glfw_KeyListener
#elif defined(SINGLE_PASS_FORWARD_PIPELINE)
class SAIGA_OPENGL_API RendererSampleWindow : public StandaloneWindow<WindowManagement::GLFW, ForwardRenderer>,
                                              public glfw_KeyListener
#endif
{
   public:
    RendererSampleWindow();
    ~RendererSampleWindow() {}

    void update(float dt) override;
    void interpolate(float dt, float interpolation) override;

    virtual void render(RenderInfo render_info) override;

    virtual void keyPressed(int key, int scancode, int mods) override;
    virtual void keyReleased(int key, int scancode, int mods) override;

   protected:
    Glfw_Camera<PerspectiveCamera> camera;
    ProceduralSkybox skybox;

    bool showSkybox = true;
};

}  // namespace Saiga
