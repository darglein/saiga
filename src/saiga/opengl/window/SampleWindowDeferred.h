/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/glfw/all.h"
#include "saiga/opengl/assets/all.h"
#include "saiga/opengl/rendering/deferredRendering/deferredRendering.h"
#include "saiga/opengl/rendering/forwardRendering/forwardRendering.h"
#include "saiga/opengl/rendering/renderer.h"
#include "saiga/opengl/window/WindowTemplate.h"
#include "saiga/opengl/window/glfw_window.h"
#include "saiga/opengl/world/LineSoup.h"
#include "saiga/opengl/world/pointCloud.h"
#include "saiga/opengl/world/proceduralSkybox.h"



namespace Saiga
{
/**
 * This is the class from which most saiga samples inherit from.
 * It's a basic scene with a camera, a skybox and a ground plane.
 *
 * @brief The SampleWindowDeferred class
 */
class SAIGA_OPENGL_API SampleWindowDeferred : public StandaloneWindow<WindowManagement::GLFW, DeferredRenderer>,
                                              public glfw_KeyListener
{
   public:
    SampleWindowDeferred();
    ~SampleWindowDeferred();

    void update(float dt) override;
    void interpolate(float dt, float interpolation) override;



    virtual void render(Camera* camera, RenderPass render_pass) override;

    virtual bool key_event(GLFWwindow* window, int key, int scancode, int action, int mods) override;
    virtual bool character_event(GLFWwindow* window, unsigned int codepoint) override { return false; }



   protected:
    std::shared_ptr<DirectionalLight> sun;
    Glfw_Camera<PerspectiveCamera> camera;

    SimpleAssetObject groundPlane;
    ProceduralSkybox skybox;

    bool showSkybox = true;
    bool showGrid   = true;
};

}  // namespace Saiga
