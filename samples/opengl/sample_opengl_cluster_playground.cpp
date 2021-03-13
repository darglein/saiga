/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/math/random.h"
#include "saiga/core/model/model_from_shape.h"
#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/opengl/window/RendererSampleWindow.h"
#include "saiga/opengl/world/skybox.h"

using namespace Saiga;

#define SEED 9

class Sample : public RendererSampleWindow
{
    using Base = RendererSampleWindow;

   public:
    Sample()
    {
        // Define GUI layout
        auto editor_layout = std::make_unique<EditorLayoutL>();
        editor_layout->RegisterImguiWindow("Rendering Lighting Sample", EditorLayoutL::WINDOW_POSITION_LEFT);
        editor_gui.SetLayout(std::move(editor_layout));

#ifdef MULTI_PASS_DEFERRED_PIPELINE
        renderer->lighting.stencilCulling = false;  // Required since stencil does limit to 256 lights.
#endif

        planeAsset  = std::make_shared<ColoredAsset>(PlaneMesh(Plane(vec3(0, -0.1, 0), vec3(0, 1, 0))));
        sponzaAsset = std::make_shared<ColoredAsset>(UnifiedModel("models/sponza.obj"));
        boxAsset    = std::make_shared<ColoredAsset>(BoxMesh(AABB(make_vec3(-0.5), make_vec3(0.5))));

        plane.asset = planeAsset;
        plane.setScale(vec3(20, 1, 20));
        plane.calculateModel();

        blockingPlane0.asset = planeAsset;
        blockingPlane0.setSimpleDirection(vec3(0, 1, 0));
        blockingPlane0.setScale(vec3(5.5f, 2, 2));
        blockingPlane0.setPosition(vec3(-6, 7, 17));
        blockingPlane0.calculateModel();

        blockingPlane1.asset = planeAsset;
        blockingPlane1.setSimpleDirection(vec3(0, 1, 0));
        blockingPlane1.setScale(vec3(5.5f, 2, 2));
        blockingPlane1.setPosition(vec3(6, 7, 17));
        blockingPlane1.calculateModel();

        sponza.asset = sponzaAsset;
        sponza.setScale(make_vec3(0.025f));
        sponza.calculateModel();

        Random::setSeed(SEED);
        for (auto& b : boxes)
        {
            b.asset = boxAsset;
            b.setScale(linearRand(vec3(4.0f, 0.25f, 4.0f), vec3(8.0f, 1.0f, 8.0f)));
            float z     = linearRand(-19.0f, 19.0f);
            float width = mix(4.0f, 44.0f, (-z + 19.0f) / 38.0f);
            float x     = linearRand(-width * 0.5f, width * 0.5f);
            b.setPosition(vec3(x, 0.5f * b.scale.y(), z));
            b.calculateModel();
        }

        int maxSize = ShaderStorageBuffer::getMaxShaderStorageBlockSize();

        int maximumNumberOfRendererSupportedDirectionalLights = maxSize / (int)sizeof(DirectionalLight::ShaderData);
        int maximumNumberOfRendererSupportedPointLights       = maxSize / (int)sizeof(PointLight::ShaderData);
        int maximumNumberOfRendererSupportedSpotLights        = maxSize / (int)sizeof(SpotLight::ShaderData);

        renderer->setLightMaxima(maximumNumberOfRendererSupportedDirectionalLights,
                                 maximumNumberOfRendererSupportedPointLights,
                                 maximumNumberOfRendererSupportedSpotLights);

#ifdef SINGLE_PASS_FORWARD_PIPELINE
        const char* shaderStr = renderer->getColoredShaderSource();

        auto deferredShader = shaderLoader.load<MVPColorShader>(shaderStr,
                                                                {{ GL_FRAGMENT_SHADER,
                                                                   "#define DEFERRED",
                                                                   1 }});
        auto depthShader = shaderLoader.load<MVPColorShader>(shaderStr, {{ GL_FRAGMENT_SHADER, "#define DEPTH", 1 }});

        ShaderPart::ShaderCodeInjections sci;
        sci.emplace_back(GL_VERTEX_SHADER, "#define FORWARD_LIT", 1);
        sci.emplace_back(GL_FRAGMENT_SHADER, "#define FORWARD_LIT", 1);

        sci.emplace_back(GL_FRAGMENT_SHADER,
                         "#define MAX_DL_COUNT" + std::to_string(maximumNumberOfRendererSupportedDirectionalLights), 2);
        sci.emplace_back(GL_FRAGMENT_SHADER,
                         "#define MAX_PL_COUNT" + std::to_string(maximumNumberOfRendererSupportedPointLights), 3);
        sci.emplace_back(GL_FRAGMENT_SHADER,
                         "#define MAX_SL_COUNT" + std::to_string(maximumNumberOfRendererSupportedSpotLights), 4);
        auto forwardShader = shaderLoader.load<MVPColorShaderFL>(shaderStr, sci);

        auto wireframeShader = shaderLoader.load<MVPColorShader>(shaderStr);

        planeAsset->setShader(deferredShader, forwardShader, depthShader, wireframeShader);
        sponzaAsset->setShader(deferredShader, forwardShader, depthShader, wireframeShader);
        boxAsset->setShader(deferredShader, forwardShader, depthShader, wireframeShader);
#endif

        currentPlayground = 3;

        setupPlayground(currentPlayground);

        std::cout << "Program Initialized!" << std::endl;
    }


    void update(float dt) override
    {
        Base::update(dt);
        switch (currentPlayground)
        {
            case 1:
            {
                for (int i = 0; i < pointLights.size(); ++i)
                {
                    vec2& ex = extras[i];
                    auto pl  = pointLights[i];
                    float h  = pl->getPosition().y();
                    ex[1] += 0.5f * dt;
                    vec2 point((ex[0] + 16.f) * cos(ex[1]), ex[0] * sin(ex[1]));
                    pl->setPosition(vec3(point.x(), h, point.y()));
                }
            }
            break;

            default:
                break;
        }
    }

    void setupPlayground(int index)
    {
        renderer->lighting.pointLights.clear();
        renderer->lighting.spotLights.clear();
        renderer->lighting.directionalLights.clear();
        pointLights.clear();
        spotLights.clear();
        extras.clear();

        switch (index)
        {
            case 0:
            {
                Random::setSeed(SEED);
                float r = 0.5f;
                for (int i = 0; i < 256; ++i)
                {
                    float theta = (float)i / 256 * two_pi<float>() * 6.f;
                    vec2 point(r * cos(theta), r * sin(theta));

                    auto light = std::make_shared<PointLight>();
                    light->setIntensity(1);
                    light->setRadius(1);
                    light->setPosition(vec3(point.x(), 0.5, point.y()));

                    light->setColorDiffuse(linearRand(vec3(0, 0, 0), vec3(1, 1, 1)));

                    renderer->lighting.AddLight(light);
                    pointLights.push_back(light);
                    r += 0.07f;
                }
                auto dl = std::make_shared<DirectionalLight>();
                dl->setIntensity(0.0f);
                dl->setAmbientIntensity(0.1f);
                renderer->lighting.AddLight(dl);

#ifndef MULTI_PASS_DEFERRED_PIPELINE
                renderer->lighting.setClusterType(0);
#endif

                camera.position = vec4(-0.125262, 30.514, 51.6262, 1);
                camera.rot      = quat(0.961825, -0.273666, 6.98492e-10, 1.60071e-09);
            }
            break;
            case 1:
            {
                Random::setSeed(SEED);
                for (int i = 0; i < 16000; ++i)
                {
                    float r     = linearRand(0.5f, 20.0f);
                    float theta = (float)i / 16000 * two_pi<float>();
                    extras.push_back(vec2(r, theta));
                    vec2 point((r + 14.f) * cos(theta), r * sin(theta));

                    auto light = std::make_shared<PointLight>();
                    light->setIntensity(1);
                    light->setRadius(linearRand(0.5f, 2.0f));
                    float h = linearRand(0.25f, 25.0f);
                    light->setPosition(vec3(point.x(), h, point.y()));

                    light->setColorDiffuse(linearRand(vec3(0, 0, 0), vec3(1, 1, 1)));

                    renderer->lighting.AddLight(light);
                    pointLights.push_back(light);
                }
                auto dl = std::make_shared<DirectionalLight>();
                dl->setIntensity(0.0f);
                dl->setAmbientIntensity(0.1f);
                renderer->lighting.AddLight(dl);

#ifndef MULTI_PASS_DEFERRED_PIPELINE
                renderer->lighting.setClusterType(2);
#endif

                camera.position = vec4(-35.3095, 16.6492, 2.20442, 1);
                camera.rot      = quat(0.732921, -0.0343305, -0.678689, -0.0318128);
            }
            break;
            case 2:
            {
                Random::setSeed(SEED);
                for (int i = 0; i < 4096; ++i)
                {
                    auto light = std::make_shared<PointLight>();
                    light->setIntensity(0.25);

                    float z     = linearRand(-19.0f, 19.0f);
                    float width = mix(4.0f, 44.0f, (-z + 19.0f) / 38.0f);
                    float x     = linearRand(-width * 0.5f, width * 0.5f);

                    light->setRadius(linearRand(0.5f, 2.0f));
                    light->setPosition(
                        linearRand(vec3(x, 0.5 * light->getRadius(), z), vec3(x, light->getRadius(), z)));

                    light->setColorDiffuse(linearRand(vec3(0, 0, 0), vec3(1, 1, 1)));

                    renderer->lighting.AddLight(light);
                    pointLights.push_back(light);
                }
                auto dl = std::make_shared<DirectionalLight>();
                dl->setIntensity(0.0f);
                dl->setAmbientIntensity(0.1f);
                renderer->lighting.AddLight(dl);

#ifndef MULTI_PASS_DEFERRED_PIPELINE
                renderer->lighting.setClusterType(2);
#endif

                camera.position = vec4(0.381537, 1.63556, 19.5328, 1);
                camera.rot      = quat(0.999195, -0.0398272, 0.00174378, 7.06406e-05);
            }
            break;
            case 3:
            {
                Random::setSeed(SEED);
                for (int i = 0; i < 4096; ++i)
                {
                    auto light = std::make_shared<PointLight>();
                    light->setIntensity(1);

                    light->setRadius(linearRand(0.5f, 2.0f));
                    light->setPosition(linearRand(vec3(-19.0f, 0.5 * light->getRadius(), -19.0f),
                                                  vec3(19.0f, light->getRadius(), 19.0f)));

                    light->setColorDiffuse(linearRand(vec3(0, 0, 0), vec3(1, 1, 1)));

                    renderer->lighting.AddLight(light);
                    pointLights.push_back(light);
                }
                auto dl = std::make_shared<DirectionalLight>();
                dl->setIntensity(0.0f);
                dl->setAmbientIntensity(0.1f);
                renderer->lighting.AddLight(dl);

                camera.position = vec4(0.123655, 9.77907, 21.8321, 1);
                camera.rot      = quat( 0.966454, -0.256838, -1.63342e-09, -2.06352e-09);


#ifndef MULTI_PASS_DEFERRED_PIPELINE
                renderer->lighting.setClusterType(2);
#endif
            }
            break;

            default:
                break;
        }
    }

    void renderPlayground(int index, Camera* camera, RenderPass render_pass)
    {
        switch (index)
        {
            case 0:
            {
                if (render_pass == RenderPass::Shadow)
                {
                    plane.renderDepth(camera);
                }
#if defined(SINGLE_PASS_DEFERRED_PIPELINE) || defined(MULTI_PASS_DEFERRED_PIPELINE)
                if (render_pass == RenderPass::Deferred)
                {
                    plane.render(camera);
                }
#elif defined(SINGLE_PASS_FORWARD_PIPELINE)
                if (render_pass == RenderPass::DepthPrepass)
                {
                    plane.renderDepth(camera);
                }
                if (render_pass == RenderPass::Forward)
                {
                    plane.renderForward(camera);
                }
#endif
            }
            break;
            case 1:
            {
                if (render_pass == RenderPass::Shadow)
                {
                    sponza.renderDepth(camera);
                }
#if defined(SINGLE_PASS_DEFERRED_PIPELINE) || defined(MULTI_PASS_DEFERRED_PIPELINE)
                if (render_pass == RenderPass::Deferred)
                {
                    sponza.render(camera);
                }
#elif defined(SINGLE_PASS_FORWARD_PIPELINE)
                if (render_pass == RenderPass::DepthPrepass)
                {
                    sponza.renderDepth(camera);
                }
                if (render_pass == RenderPass::Forward)
                {
                    sponza.renderForward(camera);
                }
#endif
            }
            break;
            case 2:
            {
                if (render_pass == RenderPass::Shadow)
                {
                    plane.renderDepth(camera);
                    for (auto& b : boxes)
                    {
                        b.renderDepth(camera);
                    }
                }
#if defined(SINGLE_PASS_DEFERRED_PIPELINE) || defined(MULTI_PASS_DEFERRED_PIPELINE)
                if (render_pass == RenderPass::Deferred)
                {
                    plane.render(camera);
                    for (auto& b : boxes)
                    {
                        b.render(camera);
                    }
                }
#elif defined(SINGLE_PASS_FORWARD_PIPELINE)
                if (render_pass == RenderPass::DepthPrepass)
                {
                    plane.renderDepth(camera);
                    for (auto& b : boxes)
                    {
                        b.renderDepth(camera);
                    }
                }
                if (render_pass == RenderPass::Forward)
                {
                    plane.renderForward(camera);
                    for (auto& b : boxes)
                    {
                        b.renderForward(camera);
                    }
                }
#endif
                break;
            }
            case 3:
            {
                if (render_pass == RenderPass::Shadow)
                {
                    blockingPlane0.renderDepth(camera);
                    blockingPlane1.renderDepth(camera);
                    plane.renderDepth(camera);
                }
#if defined(SINGLE_PASS_DEFERRED_PIPELINE) || defined(MULTI_PASS_DEFERRED_PIPELINE)
                if (render_pass == RenderPass::Deferred)
                {
                    blockingPlane0.render(camera);
                    blockingPlane1.render(camera);
                    plane.render(camera);
                }
#elif defined(SINGLE_PASS_FORWARD_PIPELINE)
                if (render_pass == RenderPass::DepthPrepass)
                {
                    blockingPlane0.renderDepth(camera);
                    blockingPlane1.renderDepth(camera);
                    plane.renderDepth(camera);
                }
                if (render_pass == RenderPass::Forward)
                {
                    blockingPlane0.renderForward(camera);
                    blockingPlane1.renderForward(camera);
                    plane.renderForward(camera);
                }
#endif
            }
            break;

            default:
                break;
        }
    }

    void render(Camera* camera, RenderPass render_pass) override
    {
        Base::render(camera, render_pass);

        renderPlayground(currentPlayground, camera, render_pass);

        if (render_pass == RenderPass::GUI)
        {
            ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Once);
            ImGui::SetNextWindowSize(ImVec2(400, 200), ImGuiCond_Once);
            ImGui::Begin("Cluster Playground Sample");

            if (ImGui::Combo("PlaygroundScene", &currentPlayground, descriptions, sceneCount))
            {
                setupPlayground(currentPlayground);
            }

            ImGui::End();
        }
    }

   private:
    std::shared_ptr<ColoredAsset> planeAsset, boxAsset, sponzaAsset;

    SimpleAssetObject plane;
    SimpleAssetObject sponza;
    std::array<SimpleAssetObject, 16> boxes;
    SimpleAssetObject blockingPlane0;
    SimpleAssetObject blockingPlane1;

    std::vector<std::shared_ptr<PointLight>> pointLights;
    std::vector<std::shared_ptr<SpotLight>> spotLights;

    std::vector<vec2> extras;

    int currentPlayground = 0;

    static const int sceneCount = 6;

    const char* descriptions[sceneCount] = {
        "SIMPLE_PLANE", "SPONZA", "USEFUL_DEPTH_SPLITS", "BLOCKED_VIEW", "PERFECT_DISTRIBUTION", "ALL_IN_ONE_SPOT"};
};

int main(const int argc, const char* argv[])
{
    initSaigaSample();
    Sample example;
    example.run();
    return 0;
}
