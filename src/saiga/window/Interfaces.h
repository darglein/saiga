/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <saiga/config.h>



namespace Saiga {

class Camera;

/**
 * Base class of all render engines.
 * This includes the deferred and forward OpenGL engines
 * as well as the Vulkan renderers.
 */
class SAIGA_GLOBAL RendererBase
{
public:
    int outputWidth = -1, outputHeight = -1;

    virtual ~RendererBase() {}

    virtual void renderImGui() {}
    virtual float getTotalRenderTime() {return 0;}

    virtual void resize(int windowWidth, int windowHeight) {}
    virtual void render(Camera *cam) = 0;
    virtual void bindCamera(Camera* cam) = 0;
};


}
