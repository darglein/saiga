/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/opengl/rendering/lighting/directional_light.h"
#include "saiga/opengl/rendering/lighting/point_light.h"
#include "saiga/opengl/rendering/lighting/spot_light.h"

#include <set>


namespace Saiga
{
class SAIGA_OPENGL_API LightManager
{
   public:
    LightManager() {}

    // This function does all the work and must be called every frame.
    // It computes culling and creates the active_xx arrays below.
    void Prepare(Camera* cam);

    void AddLight(std::shared_ptr<DirectionalLight> l) { directionalLights.insert(l); }
    void AddLight(std::shared_ptr<PointLight> l) { pointLights.insert(l); }
    void AddLight(std::shared_ptr<SpotLight> l) { spotLights.insert(l); }

    void removeLight(std::shared_ptr<DirectionalLight> l) { directionalLights.erase(l); }
    void removeLight(std::shared_ptr<PointLight> l) { pointLights.erase(l); }
    void removeLight(std::shared_ptr<SpotLight> l) { spotLights.erase(l); }

    // The active lights are all lights, which are active and not culled.
    // These vectors are recomputed every frame
    std::vector<PointLight*> active_point_lights;
    std::vector<SpotLight*> active_spot_lights;
    std::vector<DirectionalLight*> active_directional_lights;

    // The shader data arrays are recomputed every frame and are supposed to be pushed
    // into a shader storage buffer or similar.
    // This shader storage buffer is not part of this class because the 'LightManager' is independent of
    // the rendering API
    std::vector<PointLight::ShaderData> active_point_lights_data;
    std::vector<SpotLight::ShaderData> active_spot_lights_data;
    std::vector<DirectionalLight::ShaderData> active_directional_lights_data;


    std::set<std::shared_ptr<PointLight>> pointLights;
    std::set<std::shared_ptr<SpotLight>> spotLights;
    std::set<std::shared_ptr<DirectionalLight>> directionalLights;

    // Lighting Statistics
    int totalLights;
    int visibleLights;
    int renderedDepthmaps;
    int visibleVolumetricLights;

    bool renderVolumetric = false;

    void imgui();

   protected:
    bool showLightingImgui = false;
    int selected_light     = -1;
    int selecte_light_type = 0;
    std::shared_ptr<LightBase> selected_light_ptr;

};
}  // namespace Saiga
