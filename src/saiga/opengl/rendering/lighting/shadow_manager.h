/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/assert.h"
#include "saiga/core/window/Interfaces.h"
#include "saiga/opengl/framebuffer.h"
#include "saiga/opengl/rendering/lighting/directional_light.h"
#include "saiga/opengl/rendering/lighting/point_light.h"
#include "saiga/opengl/rendering/lighting/spot_light.h"
#include "saiga/opengl/texture/ArrayTexture2D.h"
#include "saiga/opengl/uniformBuffer.h"
#include "saiga/opengl/shaderStorageBuffer.h"
namespace Saiga
{

#define SHADOW_DATA_BINDING_POINT 10

// The ShadowManager is managing and rendering the shadow maps.
// It does NOT light the scene. This is done in the respective lighting stages of the rendering algorithms.
class ShadowManager
{
   public:
    ShadowManager();
    void RenderShadowMaps(Camera* view_point, RenderingInterface* renderer, ArrayView<DirectionalLight*> dls, ArrayView<PointLight*> pls,
                          ArrayView<SpotLight*> sls);

    bool backFaceShadows = false;

    // The resulting shadow maps.
    // These can be used in the light calculation shaders
    std::unique_ptr<ArrayTexture2D> cascaded_shadows;
    std::unique_ptr<ArrayTexture2D> spot_light_shadows;
    std::unique_ptr<ArrayCubeTexture> point_light_shadows;

    ShaderStorageBuffer shadow_data_spot_light;
    ShaderStorageBuffer shadow_data_point_light;
    ShaderStorageBuffer shadow_data_directional_light;

   private:
    int num_directionallight_cascades;
    int num_directionallight_shadow_maps;
    int num_pointlight_shadow_maps;
    int num_spotlight_shadow_maps;

    int current_directional_light_array_size = 0;
    int current_spot_light_array_size        = 0;
    int current_point_light_array_size       = 0;

    Framebuffer shadow_framebuffer;
    UniformBuffer shadowCameraBuffer;

    std::vector<ShadowData> shadow_data_spot_light_cpu;
    std::vector<ShadowData> shadow_data_point_light_cpu;
    std::vector<ShadowData> shadow_data_directional_light_cpu;
};

}  // namespace Saiga
