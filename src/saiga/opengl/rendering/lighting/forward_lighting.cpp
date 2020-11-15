/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "forward_lighting.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/math/random.h"
#include "saiga/opengl/rendering/lighting/box_light.h"
#include "saiga/opengl/rendering/lighting/directional_light.h"
#include "saiga/opengl/rendering/lighting/point_light.h"
#include "saiga/opengl/rendering/lighting/spot_light.h"

namespace Saiga
{
ForwardLighting::ForwardLighting() : RendererLighting()
{
    lightDataBufferPoint.createGLBuffer(nullptr, sizeof(PointLightData) * MAX_PL_COUNT, GL_DYNAMIC_DRAW);
    lightDataBufferSpot.createGLBuffer(nullptr, sizeof(SpotLightData) * MAX_SL_COUNT, GL_DYNAMIC_DRAW);
    lightDataBufferBox.createGLBuffer(nullptr, sizeof(BoxLightData) * MAX_BL_COUNT, GL_DYNAMIC_DRAW);
    lightDataBufferDirectional.createGLBuffer(nullptr, sizeof(DirectionalLightData) * MAX_DL_COUNT, GL_DYNAMIC_DRAW);
    lightInfoBuffer.createGLBuffer(nullptr, sizeof(LightInfo), GL_DYNAMIC_DRAW);
}

ForwardLighting::~ForwardLighting() {}

void ForwardLighting::initRender()
{
    RendererLighting::initRender();
    lightDataBufferPoint.bind(POINT_LIGHT_DATA_BINDING_POINT);
    lightDataBufferSpot.bind(SPOT_LIGHT_DATA_BINDING_POINT);
    lightDataBufferBox.bind(BOX_LIGHT_DATA_BINDING_POINT);
    lightDataBufferDirectional.bind(DIRECTIONAL_LIGHT_DATA_BINDING_POINT);
    lightInfoBuffer.bind(LIGHT_INFO_BINDING_POINT);
    LightInfo li;
    LightData ld;
    li.pointLightCount       = 0;
    li.spotLightCount        = 0;
    li.boxLightCount         = 0;
    li.directionalLightCount = 0;

    // Point Lights
    PointLightData glPointLight;
    for (auto pl : pointLights)
    {
        if (!pl->shouldRender()) continue;
        glPointLight.position              = make_vec4(pl->getPosition(), 0.0f);
        glPointLight.colorDiffuse          = make_vec4(pl->getColorDiffuse(), pl->getIntensity());
        glPointLight.colorSpecular         = make_vec4(pl->getColorSpecular(), 1.0f);  // specular Intensity?
        glPointLight.attenuation           = make_vec4(pl->getAttenuation(), pl->getRadius());
        ld.pointLights[li.pointLightCount] = glPointLight;
        li.pointLightCount++;
        if (li.pointLightCount >= MAX_PL_COUNT) break;  // just ignore too many lights...
    }
    lightDataBufferPoint.updateBuffer(&ld.pointLights[0], sizeof(PointLightData) * li.pointLightCount, 0);

    // Spot Lights
    SpotLightData glSpotLight;
    for (auto sl : spotLights)
    {
        if (!sl->shouldRender()) continue;
        float cosa                = cos(radians(sl->getAngle() * 0.95f));  // make border smoother
        glSpotLight.position      = make_vec4(sl->getPosition(), cosa);
        glSpotLight.colorDiffuse  = make_vec4(sl->getColorDiffuse(), sl->getIntensity());
        glSpotLight.colorSpecular = make_vec4(sl->getColorSpecular(), 1.0f);  // specular Intensity?
        glSpotLight.attenuation   = make_vec4(sl->getAttenuation(), sl->getRadius());
        glSpotLight.direction     = make_vec4(0);
        glSpotLight.direction += sl->getModelMatrix().col(1);
        ld.spotLights[li.spotLightCount] = glSpotLight;
        li.spotLightCount++;
        if (li.spotLightCount >= MAX_SL_COUNT) break;  // just ignore too many lights...
    }
    lightDataBufferSpot.updateBuffer(&ld.spotLights[0], sizeof(SpotLightData) * li.spotLightCount, 0);

    // Box Lights
    BoxLightData glBoxLight;
    for (auto bl : boxLights)
    {
        if (!bl->shouldRender()) continue;
        glBoxLight.position            = make_vec4(bl->getPosition(), 0.0f);
        glBoxLight.colorDiffuse        = make_vec4(bl->getColorDiffuse(), bl->getIntensity());
        glBoxLight.colorSpecular       = make_vec4(bl->getColorSpecular(), 1.0f);  // specular Intensity?
        ld.boxLights[li.boxLightCount] = glBoxLight;
        li.boxLightCount++;
        if (li.boxLightCount >= MAX_BL_COUNT) break;  // just ignore too many lights...
    }
    lightDataBufferBox.updateBuffer(&ld.boxLights[0], sizeof(BoxLightData) * li.boxLightCount, 0);

    // Directional Lights
    DirectionalLightData glDirectionalLight;
    for (auto dl : directionalLights)
    {
        if (!dl->shouldRender()) continue;
        glDirectionalLight.position      = make_vec4(dl->getPosition(), 0.0f);
        glDirectionalLight.colorDiffuse  = make_vec4(dl->getColorDiffuse(), dl->getIntensity());
        glDirectionalLight.colorSpecular = make_vec4(dl->getColorSpecular(), 1.0f);  // specular Intensity?
        glDirectionalLight.direction     = make_vec4(dl->getDirection(), 0.0f);
        ld.directionalLights[li.directionalLightCount] = glDirectionalLight;
        li.directionalLightCount++;
        if (li.directionalLightCount >= MAX_DL_COUNT) break;  // just ignore too many lights...
    }
    lightDataBufferDirectional.updateBuffer(&ld.directionalLights[0],
                                            sizeof(DirectionalLightData) * li.directionalLightCount, 0);


    lightInfoBuffer.updateBuffer(&li, sizeof(LightInfo), 0);
    visibleLights = li.pointLightCount + li.spotLightCount + li.boxLightCount + li.directionalLightCount;
}

void ForwardLighting::render(Camera* cam, const ViewPort& viewPort)
{
    // Does nothing
    RendererLighting::render(cam, viewPort);

    if (drawDebug)
    {
        //        glDepthMask(GL_TRUE);
        renderDebug(cam);
        //        glDepthMask(GL_FALSE);
    }
    assert_no_glerror();
}

void ForwardLighting::renderImGui(bool* p_open)
{
    RendererLighting::renderImGui();
    if (!ImGui::Begin("ForwardLighting")) return;

    int32_t count = pointLights.size();
    if (ImGui::InputInt("Point Light Count (wanted)", &count))
    {
        if (count > pointLights.size())
        {
            count -= pointLights.size();
            for (int32_t i = 0; i < count; ++i)
            {
                std::shared_ptr<PointLight> light = std::make_shared<PointLight>();
                light->setAttenuation(AttenuationPresets::Quadratic);
                light->setIntensity(1);


                light->setRadius(linearRand(2, 8));

                light->setPosition(linearRand(vec3(-16, 0.5, -16), vec3(16, light->getRadius(), 16)));

                light->setColorDiffuse(linearRand(vec3(0, 0, 0), vec3(1, 1, 1)));
                light->calculateModel();

                light->createShadowMap(512, 512, ShadowQuality::HIGH);
                light->enableShadows();
                AddLight(light);
            }
        }
        else if (count < pointLights.size())
        {
            count = pointLights.size() - count;
            for (int32_t i = 0; i < count; ++i)
            {
                pointLights.erase(--pointLights.end());
            }
        }
    }
    if (ImGui::Button("Normalize Point Lights"))
    {
        for (auto pl : pointLights)
        {
            float intensity = pl->getIntensity();
            intensity       = 1.0f / pl->getRadius();
            pl->setIntensity(intensity);
        }
    }

    count = spotLights.size();
    if (ImGui::InputInt("Spot Light Count (wanted)", &count))
    {
        if (count > spotLights.size())
        {
            count -= spotLights.size();
            for (int32_t i = 0; i < count; ++i)
            {
                std::shared_ptr<SpotLight> light = std::make_shared<SpotLight>();
                light->setAttenuation(AttenuationPresets::Quadratic);
                light->setIntensity(1);


                light->setRadius(linearRand(2, 8));

                light->setPosition(linearRand(vec3(-16, 1, -16), vec3(16, light->getRadius(), 16)));
                light->setAngle(linearRand(35, 65));

                light->setColorDiffuse(linearRand(vec3(0, 0, 0), vec3(1, 1, 1)));
                light->calculateModel();

                light->createShadowMap(512, 512, ShadowQuality::HIGH);
                light->enableShadows();
                AddLight(light);
            }
        }
        else if (count < spotLights.size())
        {
            count = spotLights.size() - count;
            for (int32_t i = 0; i < count; ++i)
            {
                spotLights.erase(--spotLights.end());
            }
        }
    }
    if (ImGui::Button("Normalize Spot Lights"))
    {
        for (auto sl : spotLights)
        {
            float intensity = sl->getIntensity();
            intensity       = 1.0f / sl->getRadius();
            sl->setIntensity(intensity);
        }
    }

    count = directionalLights.size();
    if (ImGui::InputInt("Directional Light Count (wanted)", &count))
    {
        if (count > directionalLights.size())
        {
            count -= directionalLights.size();
            for (int32_t i = 0; i < count; ++i)
            {
                std::shared_ptr<DirectionalLight> light = std::make_shared<DirectionalLight>();

                light->createShadowMap(2048, 2048, 3, ShadowQuality::LOW);
                light->enableShadows();

                light->setAmbientIntensity(0.01);

                vec3 dir = Random::sphericalRand(1).cast<float>();
                if (dir.y() > 0) dir.y() *= -1;

                light->setIntensity(0.7);
            light->setDirection(dir);
                AddLight(light);
            }
        }
        else if (count < directionalLights.size())
        {
            count = directionalLights.size() - count;
            for (int32_t i = 0; i < count; ++i)
            {
                directionalLights.erase(--directionalLights.end());
            }
        }
    }
    ImGui::End();
}

}  // namespace Saiga