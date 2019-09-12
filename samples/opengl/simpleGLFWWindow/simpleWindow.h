/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once


#include "saiga/opengl/window/SampleWindowDeferred.h"

// a
#include "saiga/core/glfw/all.h"
using namespace Saiga;

class Sample : public StandaloneWindow<WindowManagement::GLFW, DeferredRenderer>, public glfw_KeyListener
{
   public:
    Glfw_Camera<PerspectiveCamera> camera;

    SimpleAssetObject cube1, cube2;
    SimpleAssetObject groundPlane;
    SimpleAssetObject sphere;

    ProceduralSkybox skybox;

    std::shared_ptr<DirectionalLight> sun;

    Sample();
    ~Sample();

    void update(float dt) override;
    void interpolate(float dt, float interpolation) override;
    void render(Camera* cam) override;
    void renderDepth(Camera* cam) override;
    void renderOverlay(Camera* cam) override;
    void renderFinal(Camera* cam) override;

    virtual bool key_event(GLFWwindow* window, int key, int scancode, int action, int mods) override;
    virtual bool character_event(GLFWwindow* window, unsigned int codepoint) override;
};
