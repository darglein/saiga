/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "renderer_lighting.h"

#include "saiga/core/geometry/triangle_mesh_generator.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/core/math/imath.h"
#include "saiga/core/util/tostring.h"
#include "saiga/opengl/error.h"
#include "saiga/opengl/rendering/deferredRendering/deferredRendering.h"
#include "saiga/opengl/rendering/program.h"
#include "saiga/opengl/rendering/renderer.h"
#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/opengl/texture/CubeTexture.h"

namespace Saiga
{
RendererLighting::RendererLighting()
{
    createLightMeshes();
    shadowCameraBuffer.createGLBuffer(nullptr, sizeof(CameraDataGLSL), GL_DYNAMIC_DRAW);
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

    if (!boxLightShader)
    {
        boxLightShader       = shaderLoader.load<BoxLightShader>(names.boxLightShader);
        boxLightShadowShader = shaderLoader.load<BoxLightShader>(names.boxLightShader, shadowInjection);
    }
}

void RendererLighting::init(int _width, int _height, bool _useTimers)
{
    this->width  = _width;
    this->height = _height;
    useTimers    = _useTimers;

    if (useTimers)
    {
        timers2.resize(5);
        for (int i = 0; i < 5; ++i)
        {
            timers2[i].create();
        }
        timerStrings.resize(5);
        timerStrings[0] = "Init";
        timerStrings[1] = "Point Lights";
        timerStrings[2] = "Spot Lights";
        timerStrings[3] = "Box Lights";
        timerStrings[4] = "Directional Lights";
    }

    int shadowSamplesX = round(sqrt((float)shadowSamples));
    shadowInjection.emplace_back(GL_FRAGMENT_SHADER, "#define SHADOWS", 1);
    shadowInjection.emplace_back(GL_FRAGMENT_SHADER, "#define SHADOW_SAMPLES_X " + std::to_string(shadowSamplesX), 2);
}

void RendererLighting::resize(int _width, int _height)
{
    this->width  = _width;
    this->height = _height;
}

void RendererLighting::cullLights(Camera* cam)
{
    visibleLights = directionalLights.size();

    // cull lights that are not visible
    for (auto& light : spotLights)
    {
        if (light->isActive())
        {
            light->calculateCamera();
            light->shadowCamera.recalculatePlanes();
            bool visible = !light->cullLight(cam);
            visibleLights += visible;
        }
    }

    for (auto& light : boxLights)
    {
        if (light->isActive())
        {
            light->calculateCamera();
            light->shadowCamera.recalculatePlanes();
            bool visible = !light->cullLight(cam);
            visibleLights += visible;
        }
    }


    for (auto& light : pointLights)
    {
        if (light->isActive())
        {
            bool visible = !light->cullLight(cam);
            visibleLights += visible;
        }
    }
}

void RendererLighting::printTimings()
{
    if (!useTimers) return;
    for (int i = 0; i < 5; ++i)
    {
        std::cout << "\t " << getTime(i) << "ms " << timerStrings[i] << std::endl;
    }
}


void RendererLighting::initRender()
{
    totalLights       = 0;
    visibleLights     = 0;
    renderedDepthmaps = 0;
    totalLights       = directionalLights.size() + spotLights.size() + pointLights.size() + boxLights.size();
    visibleLights     = totalLights;
}

void RendererLighting::renderDepthMaps(RenderingInterface* renderer)
{
    // When GL_POLYGON_OFFSET_FILL, GL_POLYGON_OFFSET_LINE, or GL_POLYGON_OFFSET_POINT is enabled,
    // each fragment's depth value will be offset after it is interpolated from the depth values of the appropriate
    // vertices. The value of the offset is factor×DZ+r×units, where DZ is a measurement of the change in depth
    // relative to the screen area of the polygon, and r is the smallest value that is guaranteed to produce a
    // resolvable offset for a given implementation. The offset is added before the depth test is performed and
    // before the value is written into the depth buffer.
    glEnable(GL_POLYGON_OFFSET_FILL);

    float shadowMult = backFaceShadows ? -1 : 1;

    if (backFaceShadows)
        glCullFace(GL_FRONT);
    else
        glCullFace(GL_BACK);


    //        glPolygonOffset(shadowMult * shadowOffsetFactor, shadowMult * shadowOffsetUnits);

    shadowCameraBuffer.bind(CAMERA_DATA_BINDING_POINT);
    DepthFunction depthFunc = [&](Camera* cam) -> void {
        renderedDepthmaps++;
        renderer->render(cam, RenderPass::Shadow);
    };
    for (auto& light : directionalLights)
    {
        glPolygonOffset(shadowMult * light->polygon_offset.x(), shadowMult * light->polygon_offset.y());
        //        glPolygonOffset(shadowMult * shadowOffsetFactor, shadowMult * shadowOffsetUnits);
        light->renderShadowmap(depthFunc, shadowCameraBuffer);
    }
    for (auto& light : boxLights)
    {
        glPolygonOffset(shadowMult * light->polygon_offset.x(), shadowMult * light->polygon_offset.y());
        light->renderShadowmap(depthFunc, shadowCameraBuffer);
    }
    for (auto& light : spotLights)
    {
        glPolygonOffset(shadowMult * light->polygon_offset.x(), shadowMult * light->polygon_offset.y());
        light->renderShadowmap(depthFunc, shadowCameraBuffer);
    }
    for (auto& light : pointLights)
    {
        glPolygonOffset(shadowMult * light->polygon_offset.x(), shadowMult * light->polygon_offset.y());
        light->renderShadowmap(depthFunc, shadowCameraBuffer);
    }
    glCullFace(GL_BACK);
    glDisable(GL_POLYGON_OFFSET_FILL);

    glPolygonOffset(0, 0);
}

void RendererLighting::render(Camera* cam, const ViewPort& viewPort) {}


void RendererLighting::renderDebug(Camera* cam)
{
    if (!drawDebug) return;
    if (!debugShader) debugShader = shaderLoader.load<MVPColorShader>("lighting/debugmesh.glsl");

    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);

    debugShader->bind();

    // ======================= Pointlights ===================

    pointLightMesh.bind();
    // center
    for (auto& obj : pointLights)
    {
        mat4 sm    = obj->model * scale(make_vec3(0.01));
        vec4 color = obj->colorDiffuse;
        if (!obj->isActive() || !obj->isVisible())
        {
            continue;
        }
        debugShader->uploadModel(sm);
        debugShader->uploadColor(color);
        pointLightMesh.draw();
    }

    // render outline
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    for (auto& obj : pointLights)
    {
        if (!obj->isActive() || !obj->isVisible())
        {
            continue;
        }
        debugShader->uploadModel(obj->model);
        debugShader->uploadColor(obj->colorDiffuse);
        pointLightMesh.draw();
        //        }
    }
    pointLightMesh.unbind();
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);


    //==================== Spotlights ==================

    spotLightMesh.bind();
    // center
    for (auto& obj : spotLights)
    {
        mat4 sm    = obj->model * scale(make_vec3(0.01));
        vec4 color = obj->colorDiffuse;
        if (!obj->isActive() || !obj->isVisible())
        {
            continue;
        }
        debugShader->uploadModel(sm);
        debugShader->uploadColor(color);
        spotLightMesh.draw();
    }

    // render outline
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    for (auto& obj : spotLights)
    {
        if (!obj->isActive() || !obj->isVisible())
        {
            continue;
        }
        debugShader->uploadModel(obj->model);
        debugShader->uploadColor(obj->colorDiffuse);
        spotLightMesh.draw();
    }
    spotLightMesh.unbind();
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);


    //==================== Box lights ====================

    boxLightMesh.bind();
    // center
    for (auto& obj : boxLights)
    {
        mat4 sm    = obj->model * scale(make_vec3(0.05));
        vec4 color = obj->colorDiffuse;
        if (!obj->isActive() || !obj->isVisible())
        {
            // render as black if light is turned off
            color = vec4(0);
        }
        debugShader->uploadModel(sm);
        debugShader->uploadColor(color);
        boxLightMesh.draw();
    }

    // render outline
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    for (auto& obj : boxLights)
    {
        //        if(obj->isSelected()){
        debugShader->uploadModel(obj->model);
        debugShader->uploadColor(obj->colorDiffuse);
        boxLightMesh.draw();
        //        }
    }
    boxLightMesh.unbind();
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    debugShader->unbind();
}

void RendererLighting::setShader(std::shared_ptr<SpotLightShader> spotLightShader,
                                 std::shared_ptr<SpotLightShader> spotLightShadowShader)
{
    this->spotLightShader       = spotLightShader;
    this->spotLightShadowShader = spotLightShadowShader;
}

void RendererLighting::setShader(std::shared_ptr<PointLightShader> pointLightShader,
                                 std::shared_ptr<PointLightShader> pointLightShadowShader)
{
    this->pointLightShader       = pointLightShader;
    this->pointLightShadowShader = pointLightShadowShader;
}

void RendererLighting::setShader(std::shared_ptr<DirectionalLightShader> directionalLightShader,
                                 std::shared_ptr<DirectionalLightShader> directionalLightShadowShader)
{
    this->directionalLightShader       = directionalLightShader;
    this->directionalLightShadowShader = directionalLightShadowShader;
}

void RendererLighting::setShader(std::shared_ptr<BoxLightShader> boxLightShader,
                                 std::shared_ptr<BoxLightShader> boxLightShadowShader)
{
    this->boxLightShader       = boxLightShader;
    this->boxLightShadowShader = boxLightShadowShader;
}

void RendererLighting::setDebugShader(std::shared_ptr<MVPColorShader> shader)
{
    this->debugShader = shader;
}

void RendererLighting::setLightMaxima(int maxDirectionalLights, int maxPointLights, int maxSpotLights, int maxBoxLights)
{
    maxDirectionalLights = std::max(0, maxDirectionalLights);
    maxPointLights       = std::max(0, maxPointLights);
    maxSpotLights        = std::max(0, maxSpotLights);
    maxBoxLights         = std::max(0, maxBoxLights);

    maximumNumberOfDirectionalLights = maxDirectionalLights;
    maximumNumberOfPointLights       = maxPointLights;
    maximumNumberOfSpotLights        = maxSpotLights;
    maximumNumberOfBoxLights         = maxBoxLights;
}



void RendererLighting::createLightMeshes()
{
    auto qb = TriangleMeshGenerator::createFullScreenQuadMesh();
    directionalLightMesh.fromMesh(*qb);


    // the create mesh returns a sphere with outer radius of 1
    // but here we want the inner radius to be 1
    // we estimate the required outer radius with apothem of regular polygons
    float n = 4.9;
    float r = 1.0f / cos(pi<float>() / n);
    //    std::cout << "point light radius " << r << std::endl;
    Sphere s(make_vec3(0), r);
    auto sb = TriangleMeshGenerator::createMesh(s, 1);
    //    sb->createBuffers(pointLightMesh);
    pointLightMesh.fromMesh(*sb);


    Cone c(make_vec3(0), vec3(0, 1, 0), 1.0f, 1.0f);
    auto cb = TriangleMeshGenerator::createMesh(c, 10);
    //    cb->createBuffers(spotLightMesh);
    spotLightMesh.fromMesh(*cb);

    AABB box(make_vec3(-1), make_vec3(1));
    auto bb = TriangleMeshGenerator::createMesh(box);
    //    bb->createBuffers(boxLightMesh);
    boxLightMesh.fromMesh(*bb);
}


template <typename T>
static void imGuiLightBox(int id, const std::string& name, T& lights)
{
    ImGui::NewLine();
    ImGui::Separator();
    ImGui::NewLine();
    ImGui::PushID(id);
    if (ImGui::CollapsingHeader(name.c_str()))
    {
        int i = 0;
        for (auto& light : lights)
        {
            ImGui::PushID(i);
            if (ImGui::CollapsingHeader(to_string(i).c_str()))
            {
                light->renderImGui();
            }
            i++;
            ImGui::PopID();
        }
    }
    ImGui::PopID();
}

void RendererLighting::renderImGui(bool* p_open)
{
    int w = 340;
    int h = 240;
    ImGui::SetNextWindowPos(ImVec2(680, height - h), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(w, h), ImGuiCond_FirstUseEver);
    ImGui::Begin("RendererLighting", p_open);

    ImGui::Text("resolution: %dx%d", width, height);
    ImGui::Text("visibleLights/totalLights: %d/%d", visibleLights, totalLights);
    ImGui::Text("renderedDepthmaps: %d", renderedDepthmaps);
    ImGui::Text("shadowSamples: %d", shadowSamples);
    ImGui::ColorEdit4("clearColor ", &clearColor[0]);
    ImGui::Checkbox("drawDebug", &drawDebug);
    if (ImGui::Checkbox("useTimers", &useTimers) && useTimers)
    {
        timers2.resize(5);
        for (int i = 0; i < 5; ++i)
        {
            timers2[i].create();
        }
        timerStrings.resize(5);
        timerStrings[0] = "Init";
        timerStrings[1] = "Point Lights";
        timerStrings[2] = "Spot Lights";
        timerStrings[3] = "Box Lights";
        timerStrings[4] = "Directional Lights";
    }
    ImGui::Checkbox("lightDepthTest", &lightDepthTest);


    if (useTimers)
    {
        ImGui::Text("Render Time (without shadow map computation)");
        for (int i = 0; i < 5; ++i)
        {
            ImGui::Text("  %f ms %s", getTime(i), timerStrings[i].c_str());
        }
    }
    ImGui::Checkbox("backFaceShadows", &backFaceShadows);
    ImGui::InputFloat("shadowOffsetFactor", &shadowOffsetFactor, 0.1, 1);
    ImGui::InputFloat("shadowOffsetUnits", &shadowOffsetUnits, 0.1, 1);
    imGuiLightBox(0, "Directional Lights", directionalLights);
    imGuiLightBox(1, "Spot Lights", spotLights);
    imGuiLightBox(2, "Point Lights", pointLights);
    imGuiLightBox(3, "Box Lights", boxLights);

    ImGui::End();
}

}  // namespace Saiga
