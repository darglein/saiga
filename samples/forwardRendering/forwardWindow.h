/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/rendering/renderer.h"
#include "saiga/world/proceduralSkybox.h"

#include "saiga/assets/all.h"
#include "saiga/assets/objAssetLoader.h"

#include "saiga/sdl/sdl.h"
#include "saiga/rendering/forwardRendering/forwardRendering.h"

#include "saiga/world/pointCloud.h"

using namespace Saiga;




class Sample :  public Updating, public Rendering, public SDL_KeyListener
{
public:
    SDLCamera<PerspectiveCamera> camera;

    PointCloud pointCloud;
    SimpleAssetObject cube1, cube2;
    SimpleAssetObject groundPlane;
    SimpleAssetObject sphere;

    ProceduralSkybox skybox;

    std::shared_ptr<Texture> t;

    Sample(OpenGLWindow& window, Renderer& renderer);
    ~Sample();

    void update(float dt) override;
    void interpolate(float dt, float interpolation) override;
    void render(Camera *cam) override;
    void renderDepth(Camera *cam) override;
    void renderOverlay(Camera *cam) override;
    void renderFinal(Camera *cam) override;

    void keyPressed(SDL_Keysym key) override;
    void keyReleased(SDL_Keysym key) override;
};


