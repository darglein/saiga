/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"
#ifdef SAIGA_USE_GLFW

#    include "saiga/core/model/model_from_shape.h"

#    include "SampleWindowForward.h"

namespace Saiga
{
SampleWindowForward::SampleWindowForward() : StandaloneWindow("config.ini")
{
    // Define GUI layout
    auto editor_layout = std::make_unique<EditorLayoutL>();
    editor_layout->RegisterImguiWindow("Saiga Sample", EditorLayoutL::WINDOW_POSITION_LEFT);
    editor_gui.SetLayout(std::move(editor_layout));

    // create a perspective camera
    float aspect = window->getAspectRatio();
    camera.setProj(60.0f, aspect, 0.1f, 50.0f);
    camera.setView(vec3(0, 5, 10), vec3(0, 0, 0), vec3(0, 1, 0));
    camera.rotationPoint = make_vec3(0);

    // Set the camera from which view the scene is rendered
    window->setCamera(&camera);


    // This simple AssetLoader can create assets from meshes and generate some generic debug assets
    //    ObjAssetLoader assetLoader;
    //    groundPlane.asset = assetLoader.loadDebugPlaneAsset2(make_ivec2(20, 20), 1.0f, Colors::firebrick,
    //    Colors::gray);

    groundPlane.asset = std::make_shared<ColoredAsset>(
        CheckerBoardPlane(make_ivec2(20, 20), 1.0f, Colors::indianred, Colors::lightgray));
}

void SampleWindowForward::update(float dt)
{
    // Update the camera position
    if (renderer->use_keyboard_input_in_3dview) camera.update(dt);
}

void SampleWindowForward::interpolate(float dt, float interpolation)
{
    // Update the camera rotation. This could also be done in 'update' but
    // doing it in the interpolate step will reduce latency
    if (renderer->use_mouse_input_in_3dview) camera.interpolate(dt, interpolation);
}



void SampleWindowForward::render(RenderInfo render_info)
{
    if (render_info.render_pass == RenderPass::Forward)
    {
        if (showSkybox) skybox.render(render_info.camera);
        if (showGrid) groundPlane.renderForward(render_info.camera);
    }
    else if (render_info.render_pass == RenderPass::GUI)
    {
    }
}
void SampleWindowForward::keyPressed(int key, int scancode, int mods)
{
    switch (key)
    {
        case GLFW_KEY_ESCAPE:
            window->close();
            break;
        default:
            break;
    }
}

}  // namespace Saiga

#endif
