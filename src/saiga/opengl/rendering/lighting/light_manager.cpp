/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "light_manager.h"

#include "saiga/core/imgui/imgui.h"

#include "internal/noGraphicsAPI.h"

namespace Saiga
{
void LightManager::Prepare(Camera* cam)
{
    cam->recalculatePlanes();

    active_point_lights.clear();
    active_spot_lights.clear();
    active_directional_lights.clear();

    active_point_lights_data.clear();
    active_spot_lights_data.clear();
    active_directional_lights_data.clear();

    totalLights       = 0;
    visibleLights     = 0;
    renderedDepthmaps = 0;
    totalLights       = directionalLights.size() + spotLights.size() + pointLights.size();
    visibleLights     = totalLights;

    visibleLights           = directionalLights.size();
    visibleVolumetricLights = 0;

    for (auto& light : directionalLights)
    {
        if (light->active)
        {
            // (no culling needed)
            light->active_light_id = active_directional_lights.size();
            light->fitShadowToCamera(cam);
            active_directional_lights.push_back(light.get());
            active_directional_lights_data.push_back(light->GetShaderData());
        }
    }

    // cull lights that are not visible
    for (auto& light : spotLights)
    {
        if (light->active)
        {
            light->calculateCamera();
            light->shadowCamera.recalculatePlanes();
            if (!light->cullLight(cam))
            {
                light->active_light_id = active_spot_lights.size();
                active_spot_lights.push_back(light.get());
                active_spot_lights_data.push_back(light->GetShaderData());
                visibleLights += 1;
                visibleVolumetricLights += light->volumetric;
            }
        }
    }

    for (auto& light : pointLights)
    {
        if (light->active)
        {
            if (!light->cullLight(cam))
            {
                light->active_light_id = active_point_lights.size();
                active_point_lights.push_back(light.get());
                active_point_lights_data.push_back(light->GetShaderData());
                visibleLights += 1;
                visibleVolumetricLights += light->volumetric;
            }
        }
    }
    renderVolumetric = visibleVolumetricLights > 0;
}
void LightManager::imgui()
{
    if (!showLightingImgui) return;

    if (ImGui::Begin("Lighting", &showLightingImgui))
    {
        ImGui::Text("Lighting Base");
        ImGui::Text("visibleLights/totalLights: %d/%d", visibleLights, totalLights);
        ImGui::Text("renderedDepthmaps: %d", renderedDepthmaps);

        if (ImGui::ListBoxHeader("Lights", 4))
        {
            int lid = 0;
            for (auto l : directionalLights)
            {
                std::string name = "Directional Light " + std::to_string(lid);
                if (ImGui::Selectable(name.c_str(), selected_light == lid))
                {
                    selected_light     = lid;
                    selected_light_ptr = l;
                }
                lid++;
            }
            for (auto l : spotLights)
            {
                std::string name = "Spot Light " + std::to_string(lid);
                if (ImGui::Selectable(name.c_str(), selected_light == lid))
                {
                    selected_light     = lid;
                    selected_light_ptr = l;
                }
                lid++;
            }
            for (auto l : pointLights)
            {
                std::string name = "Point Light " + std::to_string(lid);
                if (ImGui::Selectable(name.c_str(), selected_light == lid))
                {
                    selected_light     = lid;
                    selected_light_ptr = l;
                }
                lid++;
            }
            ImGui::ListBoxFooter();
        }
    }
    ImGui::End();

    if (selected_light_ptr)
    {
        if (ImGui::Begin("Light Data", &showLightingImgui))
        {
            selected_light_ptr->renderImGui();
        }
        ImGui::End();
    }
}
}  // namespace Saiga
