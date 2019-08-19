/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/opengl/assets/all.h"
#include "saiga/opengl/assets/objAssetLoader.h"
#include "saiga/opengl/rendering/deferredRendering/deferredRendering.h"
#include "saiga/opengl/rendering/renderer.h"
#include "saiga/core/sdl/sdl_camera.h"
#include "saiga/core/sdl/sdl_eventhandler.h"
#include "saiga/opengl/window/sdl_window.h"
#include "saiga/opengl/world/proceduralSkybox.h"

using namespace Saiga;

class Sample : public Updating, public DeferredRenderingInterface, public SDL_KeyListener
{
   public:
    bool useAspectRatio   = true;
    float ratio           = 3;
    float errorTolerance  = 1;
    bool useQuadric       = false;
    float quadricMaxError = 0.001;
    bool useHausdorf      = false;
    float hausError       = 0.01;
    bool useNormalDev     = false;
    float normalDev       = 20;
    bool useNormalFlip    = true;
    float maxNormalDev    = 8;
    bool useRoundness     = false;
    float minRoundness    = 0.4;

    bool showReduced = false;
    bool writeToFile = false;
    bool wireframe   = true;
    SDLCamera<PerspectiveCamera> camera;

    SimpleAssetObject cube1, cube2;
    SimpleAssetObject groundPlane;
    SimpleAssetObject sphere;

    ProceduralSkybox skybox;

    std::shared_ptr<DirectionalLight> sun;
    std::shared_ptr<Texture> t;

    TriangleMesh<VertexNC, GLuint> baseMesh;
    TriangleMesh<VertexNC, GLuint> reducedMesh;

    Sample(OpenGLWindow& window, OpenGLRenderer& renderer);
    ~Sample();

    void reduce();

    void update(float dt) override;
    void interpolate(float dt, float interpolation) override;
    void render(Camera* cam) override;
    void renderDepth(Camera* cam) override;
    void renderOverlay(Camera* cam) override;
    void renderFinal(Camera* cam) override;

    void keyPressed(SDL_Keysym key) override;
    void keyReleased(SDL_Keysym key) override;
};
