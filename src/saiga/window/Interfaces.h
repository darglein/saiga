/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"



namespace Saiga
{
class Camera;
class WindowBase;
class RenderingBase;
/**
 * Base class of all render engines.
 * This includes the deferred and forward OpenGL engines
 * as well as the Vulkan renderers.
 */
class SAIGA_GLOBAL RendererBase
{
   public:
    virtual ~RendererBase() {}
    RenderingBase* rendering = nullptr;


    virtual void printTimings() {}
    void setRenderObject(RenderingBase& r) { rendering = &r; }

    virtual void renderImGui(bool* p_open = nullptr) {}
    virtual float getTotalRenderTime() { return 0; }

    virtual void resize(int windowWidth, int windowHeight) {}
    virtual void render(Camera* cam)     = 0;
    virtual void bindCamera(Camera* cam) = 0;
};

class SAIGA_GLOBAL Updating
{
   public:
    Updating(WindowBase& parent);

    virtual ~Updating() {}

    // advances the state of the program by dt. All game logic should happen here
    virtual void update(float dt) {}

    virtual void parallelUpdate(float dt) { (void)dt; }

    // interpolation between two logic steps for high fps rendering.
    // Example:
    // Game loop: constant 60 Hz
    // Render rate: around 120 Hz
    //-> The game is rendered two times per update
    //
    // We don't want to render two times the same image, so the game state should be interpolated either into the future
    // or from the past. Alpha is in the range [0,1] where 1 is equivalent to a timestep of dt
    virtual void interpolate(float dt, float alpha) {}

   protected:
    WindowBase& parentWindow;
};

class SAIGA_GLOBAL RenderingBase
{
   public:
    RenderingBase(RendererBase& parent);
    virtual ~RenderingBase() {}


   protected:
    RendererBase& parentRenderer;
};


class SAIGA_GLOBAL Rendering : public RenderingBase
{
   public:
    Rendering(RendererBase& parent) : RenderingBase(parent) {}
    virtual ~Rendering() {}

    // rendering into the gbuffer
    virtual void render(Camera* cam) {}

    // render depth maps for shadow lights
    virtual void renderDepth(Camera* cam) {}

    // forward rendering path after lighting, but before post processing
    // this could be used for transparent objects
    virtual void renderOverlay(Camera* cam) {}

    // forward rendering path after lighting and after post processing
    virtual void renderFinal(Camera* cam) {}
    // protected:
    //    RendererBase& parentRenderer;
};



}  // namespace Saiga
