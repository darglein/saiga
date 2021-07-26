/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/camera/camera.h"

namespace Saiga
{
class Camera;
class WindowBase;
class RenderingInterfaceBase;


enum class RenderPass : int
{
    Forward,
    Deferred,
    Shadow,
    DepthPrepass,
    GUI,
    Final,
    Undefined
};

enum RenderFlags : int
{
    RENDER_DEFAULT = 1,
    RENDER_UNLIT   = 2,
    RENDER_SHADOW  = 4,
};

/**
 * This struct is passed to the renderers.
 */
struct RenderInfo
{
    RenderInfo() {}
    RenderInfo(Camera* cam, RenderPass rp) : camera(cam), render_pass(rp) {}
    Camera* camera         = nullptr;
    RenderPass render_pass = RenderPass::Undefined;
    RenderFlags render_flags;
    void* render_context;
};



class SAIGA_CORE_API RenderingInterface
{
   public:
    virtual ~RenderingInterface() {}
    virtual void render(RenderInfo render_info) {}
};


/**
 * Base class of all render engines.
 * This includes the deferred and forward OpenGL engines
 * as well as the Vulkan renderers.
 *
 * Each renderer needs a fitting rendering interface.
 * For example, the deferrred renderer has the DeferredRenderingInterface.
 * An application can now derive from DeferredRenderingInterface to use
 * the deferred rendering engine.
 */
class SAIGA_CORE_API RendererBase
{
   public:
    RendererBase();
    virtual ~RendererBase() {}
    RenderingInterface* rendering = nullptr;


    void setRenderObject(RenderingInterface& r) { rendering = &r; }

    virtual void renderImgui() {}

    virtual void render(const RenderInfo& renderInfo) = 0;

    bool should_render_imgui = false;
};



/**
 * Base class for applications to make them updateable.
 * The mainloop will call the appropriate functions.
 */
class SAIGA_CORE_API Updating
{
   public:
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
};

}  // namespace Saiga
