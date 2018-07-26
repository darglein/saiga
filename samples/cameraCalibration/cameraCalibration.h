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

#include "saiga/sdl/all.h"
#include "saiga/rendering/forwardRendering/forwardRendering.h"

#include "saiga/world/pointCloud.h"

using namespace Saiga;


#ifdef SAIGA_USE_OPENNI2
#include "saiga/openni2/RGBDCameraInput.h"
#endif



class Sample :  public Updating, public ForwardRenderingInterface, public SDL_KeyListener
{
public:
    SDLCamera<PerspectiveCamera> camera;

    GLPointCloud pointCloud;
    SimpleAssetObject groundPlane;


    ProceduralSkybox skybox;

    std::shared_ptr<Texture> t;

    int capture = 0;

#ifdef SAIGA_USE_OPENNI2
    RGBDCameraInput rgbdCamera;
#endif

    Sample(OpenGLWindow& window, Renderer& renderer);
    ~Sample();

    void update(float dt) override;
    void interpolate(float dt, float interpolation) override;
    void renderOverlay(Saiga::Camera *cam) override;
    void renderFinal  (Saiga::Camera *cam) override;

    void keyPressed(SDL_Keysym key) override;
    void keyReleased(SDL_Keysym key) override;
    void captureDSLR(std::string outDir, int id);
    void captureRGBD(std::string outDir, int id);
};


