/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/rendering/renderer.h"
#include "saiga/world/proceduralSkybox.h"

#include "saiga/assets/all.h"

#include "saiga/sdl/all.h"

#include "saiga/rendering/deferredRendering/deferredRendering.h"

using namespace Saiga;

class Sample :  public Updating, public Rendering, public SDL_KeyListener
{
public:
    SDLCamera<PerspectiveCamera> camera;

    SimpleAssetObject cube1, cube2;
    SimpleAssetObject groundPlane;
    SimpleAssetObject sphere;

    ProceduralSkybox skybox;

    std::shared_ptr<DirectionalLight> sun;
    std::shared_ptr<Texture> t;

    Sample(OpenGLWindow& window, Renderer& renderer);

    void update(float dt) override;
    void interpolate(float dt, float interpolation) override;
    void render(Camera *cam) override;
    void renderDepth(Camera *cam) override;
    void renderOverlay(Camera *cam) override;
    void renderFinal(Camera *cam) override;

    void keyPressed(SDL_Keysym key) override;
    void keyReleased(SDL_Keysym key) override;
};


