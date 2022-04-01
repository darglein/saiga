/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"
#ifdef SAIGA_USE_GLFW

#    include "saiga/core/model/model_from_shape.h"

#    include "SampleWindowDeferred.h"
namespace Saiga
{
SampleWindowDeferred::SampleWindowDeferred() : StandaloneWindow("config.ini")
{
    // Define GUI layout
    auto editor_layout = std::make_unique<EditorLayoutL>();
    editor_layout->RegisterImguiWindow("Saiga Sample Base", EditorLayoutL::WINDOW_POSITION_LEFT);
    editor_layout->RegisterImguiWindow("Saiga Sample", EditorLayoutL::WINDOW_POSITION_LEFT);
    editor_gui.SetLayout(std::move(editor_layout));

    // create a perspective camera
    float aspect = window->getAspectRatio();
    camera.setProj(60.0f, aspect, 0.1f, 100.0f);
    camera.setView(vec3(0, 5, 10), vec3(0, 0, 0), vec3(0, 1, 0));
    camera.rotationPoint = make_vec3(0);

    // Set the camera from which view the scene is rendered
    window->setCamera(&camera);

    groundPlane.asset = std::make_shared<ColoredAsset>(
        CheckerBoardPlane(make_ivec2(20, 20), 1.0f, Colors::indianred, Colors::lightgray));

    // create one directional light
    sun = std::make_shared<DirectionalLight>();
    renderer->lighting.AddLight(sun);
    sun->BuildCascades(3);
    sun->castShadows = true;
}

SampleWindowDeferred::~SampleWindowDeferred() {}

void SampleWindowDeferred::update(float dt)
{
    // Update the camera position
    if (renderer->use_keyboard_input_in_3dview) camera.update(dt);
    sun->fitShadowToCamera(&camera);
}

void SampleWindowDeferred::interpolate(float dt, float interpolation)
{
    // Update the camera rotation. This could also be done in 'update' but
    // doing it in the interpolate step will reduce latency
    if (renderer->use_mouse_input_in_3dview) camera.interpolate(dt, interpolation);
    render_system.Clear();
    if (showGrid)
    {
        render_system.Add(groundPlane.asset.get(), groundPlane.model, RENDER_DEFAULT);
    }
}


void SampleWindowDeferred::render(RenderInfo render_info)
{
    render_system.Render(render_info);
    if (render_info.render_pass == RenderPass::Forward)
    {
        if (showSkybox)
        {
            skybox.sunDir = vec3(sun->getDirection());
            skybox.render(render_info.camera);
        }
    }
    else if (render_info.render_pass == RenderPass::GUI)
    {
        if (!editor_gui.enabled)
        {
            ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Once);
            ImGui::SetNextWindowSize(ImVec2(400, 200), ImGuiCond_Once);
        }
        if (ImGui::Begin("Saiga Sample Base"))
        {
            ImGui::Checkbox("showSkybox", &showSkybox);
            ImGui::Checkbox("showGrid", &showGrid);
            if (ImGui::CollapsingHeader("Camera"))
            {
                camera.imgui();
            }
            if (ImGui::CollapsingHeader("Skybox"))
            {
                skybox.imgui();
            }
        }
        ImGui::End();
    }
}

void SampleWindowDeferred::keyPressed(int key, int scancode, int mods)
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
