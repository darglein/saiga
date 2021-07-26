/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "shadow_manager.h"

#include "saiga/core/camera/camera.h"
#include "saiga/opengl/error.h"
#include "saiga/opengl/shader/basic_shaders.h"
#include "saiga/opengl/texture/CubeTexture.h"

namespace Saiga
{
ShadowManager::ShadowManager()
{
    shadow_framebuffer.create();
    shadow_framebuffer.unbind();
    shadowCameraBuffer.createGLBuffer(nullptr, sizeof(CameraDataGLSL), GL_DYNAMIC_DRAW);
}

void ShadowManager::RenderShadowMaps(Camera* view_point, RenderingInterface* renderer, ArrayView<DirectionalLight*> dls,
                                     ArrayView<PointLight*> pls, ArrayView<SpotLight*> sls)
{
    num_directionallight_cascades    = 0;
    num_directionallight_shadow_maps = 0;
    num_pointlight_shadow_maps       = 0;
    num_spotlight_shadow_maps        = 0;


    shadow_data_spot_light_cpu.clear();
    shadow_data_point_light_cpu.clear();
    shadow_data_directional_light_cpu.clear();

    for (auto& light : dls)
    {
        if (light->shouldCalculateShadowMap())
        {
            light->shadow_id = num_directionallight_cascades;
            num_directionallight_shadow_maps++;
            num_directionallight_cascades += light->getNumCascades();
        }
    }

    // cull lights that are not visible
    for (auto& light : sls)
    {
        if (light->shouldCalculateShadowMap())
        {
            light->calculateCamera();
            light->shadowCamera.recalculatePlanes();
            light->shadow_id = num_spotlight_shadow_maps;
            num_spotlight_shadow_maps += light->castShadows;
        }
    }

    for (auto& light : pls)
    {
        if (light->shouldCalculateShadowMap())
        {
            light->shadow_id = num_pointlight_shadow_maps;
            num_pointlight_shadow_maps += light->castShadows;
        }
    }

    if (current_directional_light_array_size < num_directionallight_cascades)
    {
        std::cout << "resize shadow array cascades " << num_directionallight_cascades << std::endl;
        cascaded_shadows = std::make_unique<ArrayTexture2D>();
        cascaded_shadows->create(2048, 2048, num_directionallight_cascades, GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT32,
                                 GL_UNSIGNED_INT);
        cascaded_shadows->setWrap(GL_CLAMP_TO_BORDER);
        cascaded_shadows->setBorderColor(make_vec4(1.0f));
        cascaded_shadows->setFiltering(GL_LINEAR);
        // this requires the texture sampler in the shader to be sampler2DShadow
        cascaded_shadows->setParameter(GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
        cascaded_shadows->setParameter(GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);

        shadow_data_directional_light.createGLBuffer(nullptr, sizeof(ShadowData) * num_directionallight_cascades,
                                                     GL_DYNAMIC_DRAW);

        current_directional_light_array_size = num_directionallight_cascades;
    }

    if (current_spot_light_array_size < num_spotlight_shadow_maps)
    {
        std::cout << "resize shadow array " << num_spotlight_shadow_maps << std::endl;
        spot_light_shadows = std::make_unique<ArrayTexture2D>();
        spot_light_shadows->create(512, 512, num_spotlight_shadow_maps, GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT32,
                                   GL_UNSIGNED_INT);
        spot_light_shadows->setWrap(GL_CLAMP_TO_BORDER);
        spot_light_shadows->setBorderColor(make_vec4(1.0f));
        spot_light_shadows->setFiltering(GL_LINEAR);
        // this requires the texture sampler in the shader to be sampler2DShadow
        spot_light_shadows->setParameter(GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
        spot_light_shadows->setParameter(GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);

        shadow_data_spot_light.createGLBuffer(nullptr, sizeof(ShadowData) * num_spotlight_shadow_maps, GL_DYNAMIC_DRAW);

        current_spot_light_array_size = num_spotlight_shadow_maps;
    }

    if (current_point_light_array_size < num_pointlight_shadow_maps)
    {
        std::cout << "resize shadow array point" << num_pointlight_shadow_maps << std::endl;

        point_light_shadows = std::make_unique<ArrayCubeTexture>();
        point_light_shadows->create(512, 512, num_pointlight_shadow_maps * 6, GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT32,
                                    GL_UNSIGNED_INT);
        point_light_shadows->setWrap(GL_CLAMP_TO_BORDER);
        point_light_shadows->setBorderColor(make_vec4(1.0f));
        point_light_shadows->setFiltering(GL_LINEAR);
        // this requires the texture sampler in the shader to be sampler2DShadow
        point_light_shadows->setParameter(GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
        point_light_shadows->setParameter(GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);

        shadow_data_point_light.createGLBuffer(nullptr, sizeof(ShadowData) * num_pointlight_shadow_maps,
                                               GL_DYNAMIC_DRAW);

        current_point_light_array_size = num_pointlight_shadow_maps;
    }

    glEnable(GL_POLYGON_OFFSET_FILL);

    float shadowMult = backFaceShadows ? -1 : 1;

    if (backFaceShadows)
        glCullFace(GL_FRONT);
    else
        glCullFace(GL_BACK);

    shadowCameraBuffer.bind(CAMERA_DATA_BINDING_POINT);

    DepthFunction depthFunc = [&](Camera* cam) -> void {
        RenderInfo ri;
        ri.camera      = cam;
        ri.render_pass = RenderPass::Shadow;
        renderer->render(ri);
    };

    shadow_framebuffer.bind();
    for (auto& light : dls)
    {
        if (light->shouldCalculateShadowMap())
        {
            glPolygonOffset(shadowMult * light->polygon_offset.x(), shadowMult * light->polygon_offset.y());


            for (int i = 0; i < light->getNumCascades(); ++i)
            {
                glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, cascaded_shadows->getId(), 0,
                                          light->shadow_id + i);
                glViewport(0, 0, 2048, 2048);
                glClear(GL_DEPTH_BUFFER_BIT);
                glEnable(GL_DEPTH_TEST);
                glDepthMask(GL_TRUE);

                light->shadowCamera.setProj(light->orthoBoxes[i]);
                light->shadowCamera.recalculatePlanes();
                CameraDataGLSL cd(&light->shadowCamera);
                shadowCameraBuffer.updateBuffer(&cd, sizeof(CameraDataGLSL), 0);
                depthFunc(&light->shadowCamera);
                shadow_data_directional_light_cpu.push_back(light->GetShadowData(view_point));
            }
        }
    }


    for (auto& light : sls)
    {
        if (light->shouldCalculateShadowMap())
        {
            glPolygonOffset(shadowMult * light->polygon_offset.x(), shadowMult * light->polygon_offset.y());
            SAIGA_ASSERT(spot_light_shadows);
            SAIGA_ASSERT(light->shadow_id >= 0 && light->shadow_id < current_spot_light_array_size);
            glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, spot_light_shadows->getId(), 0,
                                      light->shadow_id);

            glClear(GL_DEPTH_BUFFER_BIT);
            glEnable(GL_DEPTH_TEST);
            glDepthMask(GL_TRUE);
            glViewport(0, 0, 512, 512);

            CameraDataGLSL cd(&light->shadowCamera);
            shadowCameraBuffer.updateBuffer(&cd, sizeof(CameraDataGLSL), 0);
            depthFunc(&light->shadowCamera);
            shadow_data_spot_light_cpu.push_back(light->GetShadowData(view_point));
        }
    }
    for (auto& light : pls)
    {
        if (light->shouldCalculateShadowMap())
        {
            SAIGA_ASSERT(point_light_shadows);
            SAIGA_ASSERT(light->shadow_id >= 0 && light->shadow_id < current_point_light_array_size);

            glPolygonOffset(shadowMult * light->polygon_offset.x(), shadowMult * light->polygon_offset.y());

            for (int i = 0; i < 6; i++)
            {
                glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, point_light_shadows->getId(), 0,
                                          light->shadow_id * 6 + i);

                glClear(GL_DEPTH_BUFFER_BIT);
                glEnable(GL_DEPTH_TEST);
                glDepthMask(GL_TRUE);
                glViewport(0, 0, 512, 512);

                light->calculateCamera(i);
                light->shadowCamera.recalculatePlanes();
                CameraDataGLSL cd(&light->shadowCamera);
                shadowCameraBuffer.updateBuffer(&cd, sizeof(CameraDataGLSL), 0);
                depthFunc(&light->shadowCamera);
            }
            shadow_data_point_light_cpu.push_back(light->GetShadowData(view_point));
        }
    }
    shadow_framebuffer.unbind();

    shadow_data_spot_light.updateBuffer(shadow_data_spot_light_cpu.data(),
                                        shadow_data_spot_light_cpu.size() * sizeof(ShadowData), 0);
    shadow_data_point_light.updateBuffer(shadow_data_point_light_cpu.data(),
                                         shadow_data_point_light_cpu.size() * sizeof(ShadowData), 0);
    shadow_data_directional_light.updateBuffer(shadow_data_directional_light_cpu.data(),
                                               shadow_data_directional_light_cpu.size() * sizeof(ShadowData), 0);


    glPolygonOffset(0, 0);
}


}  // namespace Saiga
