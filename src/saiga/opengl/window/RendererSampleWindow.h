/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#ifdef SAIGA_USE_SDL

#    include "saiga/core/sdl/all.h"
#    include "saiga/opengl/assets/all.h"
#    include "saiga/opengl/rendering/deferredRendering/uberDeferredRendering.h"
#    include "saiga/opengl/rendering/renderer.h"
#    include "saiga/opengl/rendering/forwardRendering/forwardRendering.h"
#    include "saiga/opengl/window/WindowTemplate.h"
#    include "saiga/opengl/window/sdl_window.h"
#    include "saiga/opengl/world/proceduralSkybox.h"



namespace Saiga
{
/**
 * This is the class from which saiga samples can inherit from.
 * It's a basic scene with a camera and a skybox.
 *
 * @brief The RendererSampleWindow class
 */

//#define SINGLE_PASS_DEFERRED_PIPELINE
#define SINGLE_PASS_FORWARD_PIPELINE

#ifdef SINGLE_PASS_DEFERRED_PIPELINE
#undef SINGLE_PASS_FORWARD_PIPELINE
class SAIGA_OPENGL_API RendererSampleWindow : public StandaloneWindow<WindowManagement::SDL, UberDeferredRenderer>,
                                              public SDL_KeyListener
#elif defined(SINGLE_PASS_FORWARD_PIPELINE)
class SAIGA_OPENGL_API RendererSampleWindow : public StandaloneWindow<WindowManagement::SDL, ForwardRenderer>,
                                              public SDL_KeyListener
#endif
{
   public:
    RendererSampleWindow();
    ~RendererSampleWindow() {}

    void update(float dt) override;
    void interpolate(float dt, float interpolation) override;

    virtual void render(Camera* camera, RenderPass render_pass) override;

    void keyPressed(SDL_Keysym key) override;
    void keyReleased(SDL_Keysym key) override;

   protected:
    SDLCamera<PerspectiveCamera> camera;
    ProceduralSkybox skybox;

    bool showSkybox = true;
};

}  // namespace Saiga

#endif
