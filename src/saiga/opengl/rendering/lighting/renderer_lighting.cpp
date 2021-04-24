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
    shadowCameraBuffer.createGLBuffer(nullptr, sizeof(CameraDataGLSL), GL_DYNAMIC_DRAW);

    shadow_framebuffer.create();
    shadow_framebuffer.unbind();

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
    cam->recalculatePlanes();


    num_directionallight_cascades    = 0;
    num_directionallight_shadow_maps = 0;
    num_pointlight_shadow_maps       = 0;
    num_spotlight_shadow_maps        = 0;


    visibleLights           = directionalLights.size();
    visibleVolumetricLights = 0;

    for (auto& light : directionalLights)
    {
        if (light->active && light->castShadows)
        {
            light->fitShadowToCamera(cam);
            light->shadow_id = num_directionallight_shadow_maps;
            num_directionallight_shadow_maps++;
            light->cascade_offset = num_directionallight_cascades;
            num_directionallight_cascades += light->getNumCascades();
        }
    }

    // cull lights that are not visible
    for (auto& light : spotLights)
    {
        if (light->active)
        {
            light->calculateCamera();
            light->shadowCamera.recalculatePlanes();
            bool visible = !light->cullLight(cam);
            visibleLights += visible;
            visibleVolumetricLights += (visible && light->volumetric);
            light->shadow_id = num_spotlight_shadow_maps;
            num_spotlight_shadow_maps += light->castShadows;
        }
    }

    for (auto& light : pointLights)
    {
        if (light->active)
        {
            bool visible = !light->cullLight(cam);
            visibleLights += visible;
            visibleVolumetricLights += (visible && light->volumetric);
            light->shadow_id = num_pointlight_shadow_maps;
            num_pointlight_shadow_maps += light->castShadows;
        }
    }

    renderVolumetric = visibleVolumetricLights > 0;
}

void RendererLighting::initRender()
{
    totalLights       = 0;
    visibleLights     = 0;
    renderedDepthmaps = 0;
    totalLights       = directionalLights.size() + spotLights.size() + pointLights.size();
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
        current_point_light_array_size = num_pointlight_shadow_maps;
    }

    for (auto& light : directionalLights)
    {
        if (light->shouldCalculateShadowMap())
        {
            glPolygonOffset(shadowMult * light->polygon_offset.x(), shadowMult * light->polygon_offset.y());
            shadow_framebuffer.bind();

            for (int i = 0; i < light->getNumCascades(); ++i)
            {
                glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, cascaded_shadows->getId(), 0,
                                          light->cascade_offset + i);
                glViewport(0, 0, 2048, 2048);
                glClear(GL_DEPTH_BUFFER_BIT);
                glEnable(GL_DEPTH_TEST);
                glDepthMask(GL_TRUE);
                shadow_framebuffer.check();

                light->shadowCamera.setProj(light->orthoBoxes[i]);
                light->shadowCamera.recalculatePlanes();
                CameraDataGLSL cd(&light->shadowCamera);
                shadowCameraBuffer.updateBuffer(&cd, sizeof(CameraDataGLSL), 0);
                depthFunc(&light->shadowCamera);

            }

            shadow_framebuffer.unbind();
        }
    }


    for (auto& light : spotLights)
    {
        if (light->shouldCalculateShadowMap())
        {
            glPolygonOffset(shadowMult * light->polygon_offset.x(), shadowMult * light->polygon_offset.y());

            shadow_framebuffer.bind();
            SAIGA_ASSERT(spot_light_shadows);
            SAIGA_ASSERT(light->shadow_id >= 0 && light->shadow_id < current_spot_light_array_size);
            glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, spot_light_shadows->getId(), 0,
                                      light->shadow_id);
            //            shadow_framebuffer.check();

            glClear(GL_DEPTH_BUFFER_BIT);
            glEnable(GL_DEPTH_TEST);
            glDepthMask(GL_TRUE);
            glViewport(0, 0, 512, 512);


            CameraDataGLSL cd(&light->shadowCamera);
            shadowCameraBuffer.updateBuffer(&cd, sizeof(CameraDataGLSL), 0);
            depthFunc(&light->shadowCamera);
            shadow_framebuffer.unbind();
            //            exit(0);

            // light->renderShadowmap(depthFunc, shadowCameraBuffer);
        }
    }
    for (auto& light : pointLights)
    {
        if (light->shouldCalculateShadowMap())
        {
            SAIGA_ASSERT(point_light_shadows);
            SAIGA_ASSERT(light->shadow_id >= 0 && light->shadow_id < current_point_light_array_size);

            glPolygonOffset(shadowMult * light->polygon_offset.x(), shadowMult * light->polygon_offset.y());

            shadow_framebuffer.bind();



            for (int i = 0; i < 6; i++)
            {
                glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, point_light_shadows->getId(), 0,
                                          light->shadow_id * 6 + i);

                glClear(GL_DEPTH_BUFFER_BIT);
                glEnable(GL_DEPTH_TEST);
                glDepthMask(GL_TRUE);
                glViewport(0, 0, 512, 512);
                shadow_framebuffer.check();

                light->calculateCamera(i);
                light->shadowCamera.recalculatePlanes();
                CameraDataGLSL cd(&light->shadowCamera);
                shadowCameraBuffer.updateBuffer(&cd, sizeof(CameraDataGLSL), 0);
                depthFunc(&light->shadowCamera);
            }
            shadow_framebuffer.unbind();
        }
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
    glDisable(GL_CULL_FACE);


    debugShader->bind();

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
    glEnable(GL_CULL_FACE);
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


void RendererLighting::setDebugShader(std::shared_ptr<MVPColorShader> shader)
{
    this->debugShader = shader;
}

void RendererLighting::setLightMaxima(int maxDirectionalLights, int maxPointLights, int maxSpotLights)
{
    maxDirectionalLights = std::max(0, maxDirectionalLights);
    maxPointLights       = std::max(0, maxPointLights);
    maxSpotLights        = std::max(0, maxSpotLights);

    maximumNumberOfDirectionalLights = maxDirectionalLights;
    maximumNumberOfPointLights       = maxPointLights;
    maximumNumberOfSpotLights        = maxSpotLights;
}



void RendererLighting::createLightMeshes()
{
    //    auto qb = TriangleMeshGenerator::createFullScreenQuadMesh();
    //    directionalLightMesh.fromMesh(*qb);

    directionalLightMesh.fromMesh(FullScreenQuad());


    // the create mesh returns a sphere with outer radius of 1
    // but here we want the inner radius to be 1
    // we estimate the required outer radius with apothem of regular polygons
    float n = 4.9;
    float r = 1.0f / cos(pi<float>() / n);
    //    std::cout << "point light radius " << r << std::endl;
    Sphere s(make_vec3(0), r);
    //    auto sb = TriangleMeshGenerator::IcoSphereMesh(s, 1);
    //    sb->createBuffers(pointLightMesh);
    //    pointLightMesh.fromMesh(*sb);
    pointLightMesh.fromMesh(IcoSphereMesh(s, 1));


    Cone c(make_vec3(0), vec3(0, 0, -1), 1.0f, 1.0f);
    //    auto cb = TriangleMeshGenerator::ConeMesh(c, 10);
    auto model = ConeMesh(c, 10);

    //    cb->createBuffers(spotLightMesh);
    spotLightMesh.fromMesh(model);
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

void RendererLighting::renderImGui()
{
    if (!showLightingImgui) return;

    if (!editor_gui.enabled)
    {
        int w = 340;
        int h = 240;
        ImGui::SetNextWindowPos(ImVec2(680, height - h), ImGuiCond_Once);
        ImGui::SetNextWindowSize(ImVec2(w, h), ImGuiCond_Once);
    }
    if (ImGui::Begin("Lighting", &showLightingImgui))
    {
        ImGui::Text("Lighting Base");
        ImGui::Text("resolution: %dx%d", width, height);
        ImGui::Text("visibleLights/totalLights: %d/%d", visibleLights, totalLights);
        ImGui::Text("renderedDepthmaps: %d", renderedDepthmaps);
        ImGui::Text("shadowSamples: %d", shadowSamples);
        ImGui::ColorEdit4("clearColor ", &clearColor[0]);
        ImGui::Checkbox("drawDebug", &drawDebug);

        ImGui::Checkbox("lightDepthTest", &lightDepthTest);


        ImGui::Checkbox("backFaceShadows", &backFaceShadows);
        ImGui::InputFloat("shadowOffsetFactor", &shadowOffsetFactor, 0.1, 1);
        ImGui::InputFloat("shadowOffsetUnits", &shadowOffsetUnits, 0.1, 1);


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
