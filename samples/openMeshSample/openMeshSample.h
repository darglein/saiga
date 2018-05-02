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

#include "saiga/rendering/lighting/directional_light.h"

using namespace Saiga;

class SimpleWindow : public Program, public SDL_KeyListener
{
public:
    bool useAspectRatio = true; float ratio = 3; float errorTolerance = 1;
    bool useQuadric = false; float quadricMaxError = 0.001;
    bool useHausdorf = false; float hausError = 0.01;
    bool useNormalDev = false; float normalDev = 20;
    bool useNormalFlip = true; float maxNormalDev = 8;
    bool useRoundness = false; float minRoundness = 0.4;

    bool showReduced = false;
    bool writeToFile = false;
    bool wireframe = true;
    SDLCamera<PerspectiveCamera> camera;

    SimpleAssetObject cube1, cube2;
    SimpleAssetObject groundPlane;
    SimpleAssetObject sphere;

    ProceduralSkybox skybox;

    std::shared_ptr<DirectionalLight> sun;
    std::shared_ptr<Texture> t;

    TriangleMesh<VertexNC,GLuint> baseMesh;
    TriangleMesh<VertexNC,GLuint> reducedMesh;

    SimpleWindow(OpenGLWindow* window);
    ~SimpleWindow();

    void reduce();

    void update(float dt) override;
    void interpolate(float dt, float interpolation) override;
    void render(Camera *cam) override;
    void renderDepth(Camera *cam) override;
    void renderOverlay(Camera *cam) override;
    void renderFinal(Camera *cam) override;

    void keyPressed(SDL_Keysym key) override;
    void keyReleased(SDL_Keysym key) override;
};


