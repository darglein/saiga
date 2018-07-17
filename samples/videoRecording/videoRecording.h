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

#include "saiga/sdl/sdl_eventhandler.h"
#include "saiga/sdl/sdl_camera.h"
#include "saiga/sdl/sdl_window.h"

#include "saiga/rendering/overlay/deferredDebugOverlay.h"


#include "saiga/ffmpeg/ffmpegEncoder.h"
#include "saiga/ffmpeg/videoEncoder.h"
#include "saiga/rendering/deferredRendering/deferredRendering.h"
#include "saiga/animation/cameraAnimation.h"

using namespace Saiga;


class Sample : public Updating, public Rendering, public SDL_KeyListener
{
public:
    SDLCamera<PerspectiveCamera> camera;

    SimpleAssetObject cube1, cube2;
    SimpleAssetObject groundPlane;
    SimpleAssetObject sphere;

    ProceduralSkybox skybox;



    std::shared_ptr<DirectionalLight> sun;

    int remainingFrames;
    bool rotateCamera = false;
    int frame = 0;
    int frameSkip = 0;
    std::shared_ptr<FFMPEGEncoder> encoder;

    VideoEncoder enc;



    Interpolation cameraInterpolation;

    Sample(OpenGLWindow& window, Renderer& renderer);
    ~Sample();

    void testBspline();

    void update(float dt) override;
    void interpolate(float dt, float interpolation) override;
    void render(Camera *cam) override;
    void renderDepth(Camera *cam) override;
    void renderOverlay(Camera *cam) override;
    void renderFinal(Camera *cam) override;

    void keyPressed(SDL_Keysym key) override;
    void keyReleased(SDL_Keysym key) override;
};


