/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "renderer_lighting.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/math/imath.h"
#include "saiga/core/model/model_from_shape.h"
#include "saiga/core/util/tostring.h"
#include "saiga/opengl/error.h"
#include "saiga/opengl/rendering/lighting/deferred_light_shader.h"
#include "saiga/opengl/rendering/program.h"
#include "saiga/opengl/rendering/renderer.h"
#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/opengl/texture/CubeTexture.h"

namespace Saiga
{
RendererLighting::RendererLighting(GLTimerSystem* timer) : timer(timer)
{
    createLightMeshes();


    main_menu.AddItem(
        "Saiga", "Lighting", [this]() { showLightingImgui = !showLightingImgui; }, 297, "F8");
}

RendererLighting::~RendererLighting() {}

void RendererLighting::loadShaders()
{
    const RendererLightingShaderNames& names = RendererLightingShaderNames();

    if (!directionalLightShader)
    {
        directionalLightShader = shaderLoader.load<DirectionalLightShader>(names.directionalLightShader);
        directionalLightShadowShader =
            shaderLoader.load<DirectionalLightShader>(names.directionalLightShader, shadowInjection);
    }

    if (!pointLightShader)
    {
        pointLightShader       = shaderLoader.load<PointLightShader>(names.pointLightShader);
        pointLightShadowShader = shaderLoader.load<PointLightShader>(names.pointLightShader, shadowInjection);
    }

    if (!spotLightShader)
    {
        spotLightShader       = shaderLoader.load<SpotLightShader>(names.spotLightShader);
        spotLightShadowShader = shaderLoader.load<SpotLightShader>(names.spotLightShader, shadowInjection);
    }
}

void RendererLighting::init(int _width, int _height, bool _useTimers)
{
    this->width  = _width;
    this->height = _height;


    int shadowSamplesX = round(sqrt((float)shadowSamples));
    shadowInjection.emplace_back(GL_FRAGMENT_SHADER, "#define SHADOWS", 1);
    shadowInjection.emplace_back(GL_FRAGMENT_SHADER, "#define SHADOW_SAMPLES_X " + std::to_string(shadowSamplesX), 2);


}

void RendererLighting::resize(int _width, int _height)
{
    this->width  = _width;
    this->height = _height;
}

void RendererLighting::ComputeCullingAndStatistics(Camera* cam)
{
    Prepare(cam);

    if (active_point_lights_data.size() > point_light_data.Size())
    {
        point_light_data.create(active_point_lights_data, GL_DYNAMIC_DRAW);
        point_light_data.bind(POINT_LIGHT_DATA_BINDING_POINT);
    }
    else
    {
        point_light_data.update(active_point_lights_data);
    }

    if (active_spot_lights_data.size() > spot_light_data.Size())
    {
        spot_light_data.create(active_spot_lights_data, GL_DYNAMIC_DRAW);
        spot_light_data.bind(SPOT_LIGHT_DATA_BINDING_POINT);
    }
    else
    {
        spot_light_data.update(active_spot_lights_data);
    }

    if (active_directional_lights_data.size() > directional_light_data.Size())
    {
        directional_light_data.create(active_directional_lights_data, GL_DYNAMIC_DRAW);
        directional_light_data.bind(DIRECTIONAL_LIGHT_DATA_BINDING_POINT);
    }
    else
    {
        directional_light_data.update(active_directional_lights_data);
    }
}

void RendererLighting::initRender() {}

void RendererLighting::renderDepthMaps(Camera* camera, RenderingInterface* renderer)
{
    std::vector<PointLight*> pls;
    for (auto p : pointLights) pls.push_back(p.get());
    std::vector<SpotLight*> sls;
    for (auto p : spotLights) sls.push_back(p.get());
    std::vector<DirectionalLight*> dls;
    for (auto p : directionalLights) dls.push_back(p.get());
    shadowManager.RenderShadowMaps(camera, renderer, dls, pls, sls);
}

void RendererLighting::render(Camera* cam, const ViewPort& viewPort) {}


void RendererLighting::renderDebug(Camera* cam)
{
    if (!drawDebug) return;
    if (!debugShader) debugShader = shaderLoader.load<MVPColorShader>("lighting/debugmesh.glsl");

    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);


    if(debugShader->bind())
    {
        // ======================= Pointlights ===================

        pointLightMesh.bind();
        // center
        for (auto& obj : pointLights)
        {
            if (!obj->active || !obj->visible)
            {
                continue;
            }
            float s = 1.f / obj->getRadius() * 0.1;
            debugShader->uploadModel(obj->ModelMatrix() * scale(make_vec3(s)));
            debugShader->uploadColor(make_vec4(obj->colorDiffuse, 1));
            pointLightMesh.draw();
        }

        // render outline
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        for (auto& obj : pointLights)
        {
            if (!obj->active || !obj->visible)
            {
                continue;
            }
            debugShader->uploadModel(obj->ModelMatrix());
            debugShader->uploadColor(make_vec4(obj->colorDiffuse, 1));
            pointLightMesh.draw();
        }
        pointLightMesh.unbind();
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);


        //==================== Spotlights ==================

        spotLightMesh.bind();
        // center
        for (auto& obj : spotLights)
        {
            if (!obj->active || !obj->visible)
            {
                continue;
            }
            float s = 1.f / obj->getRadius() * 0.1;
            debugShader->uploadModel(obj->ModelMatrix() * scale(make_vec3(s)));
            debugShader->uploadColor(make_vec4(obj->colorDiffuse, 1));
            spotLightMesh.draw();
        }

        // render outline
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        for (auto& obj : spotLights)
        {
            if (!obj->active || !obj->visible)
            {
                continue;
            }
            debugShader->uploadModel(obj->ModelMatrix());
            debugShader->uploadColor(make_vec4(obj->colorDiffuse, 1));
            spotLightMesh.draw();
        }
        spotLightMesh.unbind();
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);



        debugShader->unbind();
    }
    glEnable(GL_CULL_FACE);
}

void RendererLighting::createLightMeshes()
{
    directionalLightMesh.fromMesh(FullScreenQuad());

    // the create mesh returns a sphere with outer radius of 1
    // but here we want the inner radius to be 1
    // we estimate the required outer radius with apothem of regular polygons
    float n = 4.9;
    float r = 1.0f / cos(pi<float>() / n);
    Sphere s(make_vec3(0), r);
    pointLightMesh.fromMesh(IcoSphereMesh(s, 1));

    Cone c(make_vec3(0), vec3(0, 0, -1), 1.0f, 1.0f);
    spotLightMesh.fromMesh(ConeMesh(c, 10));
}


void RendererLighting::renderImGui()
{
    if (!showLightingImgui) return;

    if (ImGui::Begin("Lighting", &showLightingImgui))
    {
        ImGui::Text("Lighting Base");
        ImGui::Text("resolution: %dx%d", width, height);
        ImGui::Text("shadowSamples: %d", shadowSamples);
        ImGui::ColorEdit4("clearColor ", &clearColor[0]);
        ImGui::Checkbox("drawDebug", &drawDebug);

        ImGui::Checkbox("lightDepthTest", &lightDepthTest);
    }
    ImGui::End();


    LightManager::imgui();
}

}  // namespace Saiga
