/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/math/random.h"
#include "saiga/opengl/assets/objAssetLoader.h"
#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/opengl/window/RendererSampleWindow.h"
#include "saiga/opengl/world/skybox.h"


using namespace Saiga;

class Sample : public RendererSampleWindow
{
    using Base                                            = RendererSampleWindow;
    int maximumNumberOfRendererSupportedDirectionalLights = 256;
    int maximumNumberOfRendererSupportedPointLights       = 256;
    int maximumNumberOfRendererSupportedSpotLights        = 256;
    int maximumNumberOfRendererSupportedBoxLights         = 256;

   public:
    Sample()
    {
        ObjAssetLoader assetLoader;
        auto showAsset = assetLoader.loadColoredAsset("show_model.obj");

        show.asset = showAsset;
        // show.multScale(make_vec3(0.01f));
        // show.calculateModel();

        int maxSize = ShaderStorageBuffer::getMaxShaderStorageBlockSize();

        maximumNumberOfRendererSupportedDirectionalLights = std::clamp(
            maximumNumberOfRendererSupportedDirectionalLights, 0, maxSize / (int)sizeof(uber::DirectionalLightData));
        maximumNumberOfRendererSupportedPointLights =
            std::clamp(maximumNumberOfRendererSupportedPointLights, 0, maxSize / (int)sizeof(uber::PointLightData));
        maximumNumberOfRendererSupportedSpotLights =
            std::clamp(maximumNumberOfRendererSupportedSpotLights, 0, maxSize / (int)sizeof(uber::SpotLightData));
        maximumNumberOfRendererSupportedBoxLights =
            std::clamp(maximumNumberOfRendererSupportedBoxLights, 0, maxSize / (int)sizeof(uber::BoxLightData));

// Next is needed for forward.
#ifdef SINGLE_PASS_FORWARD_PIPELINE
        const char* shaderStr = renderer->getColoredShaderSource();

        renderer->setLightMaxima(maximumNumberOfRendererSupportedDirectionalLights,
                                 maximumNumberOfRendererSupportedPointLights,
                                 maximumNumberOfRendererSupportedSpotLights, maximumNumberOfRendererSupportedBoxLights);

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
        sci.emplace_back(GL_FRAGMENT_SHADER,
                         "#define MAX_BL_COUNT" + std::to_string(maximumNumberOfRendererSupportedBoxLights), 5);
        auto forwardShader = shaderLoader.load<MVPColorShaderFL>(shaderStr, sci);

        auto wireframeShader = shaderLoader.load<MVPColorShader>(shaderStr);

        showAsset->setShader(deferredShader, forwardShader, depthShader, wireframeShader);
#endif

        std::cout << "Program Initialized!" << std::endl;
    }



    void render(Camera* camera, RenderPass render_pass) override
    {
        Base::render(camera, render_pass);

#ifdef SINGLE_PASS_DEFERRED_PIPELINE
        if (render_pass == RenderPass::Deferred || render_pass == RenderPass::Shadow)
        {
            show.render(camera);
        }
#elif defined(SINGLE_PASS_FORWARD_PIPELINE)
        if (render_pass == RenderPass::Forward)
        {
            show.renderForward(camera);
        }
#endif
        if (render_pass == RenderPass::GUI)
        {
            if (!ImGui::Begin("Rendering Lighting Sample")) return;

            bool supportChanged =
                ImGui::InputInt("Renderer Supported Point Lights", &maximumNumberOfRendererSupportedPointLights);
            supportChanged |=
                ImGui::InputInt("Renderer Supported Spot Lights", &maximumNumberOfRendererSupportedSpotLights);
            supportChanged |=
                ImGui::InputInt("Renderer Supported Box Lights", &maximumNumberOfRendererSupportedBoxLights);
            supportChanged |= ImGui::InputInt("Renderer Supported Directional Lights",
                                              &maximumNumberOfRendererSupportedDirectionalLights);


            if (supportChanged)
            {
                int maxSize = ShaderStorageBuffer::getMaxShaderStorageBlockSize();

                maximumNumberOfRendererSupportedDirectionalLights =
                    std::clamp(maximumNumberOfRendererSupportedDirectionalLights, 0,
                               maxSize / (int)sizeof(uber::DirectionalLightData));
                maximumNumberOfRendererSupportedPointLights = std::clamp(maximumNumberOfRendererSupportedPointLights, 0,
                                                                         maxSize / (int)sizeof(uber::PointLightData));
                maximumNumberOfRendererSupportedSpotLights  = std::clamp(maximumNumberOfRendererSupportedSpotLights, 0,
                                                                        maxSize / (int)sizeof(uber::SpotLightData));
                maximumNumberOfRendererSupportedBoxLights =
                    std::clamp(maximumNumberOfRendererSupportedBoxLights, 0, maxSize / (int)sizeof(uber::BoxLightData));

                renderer->setLightMaxima(
                    maximumNumberOfRendererSupportedDirectionalLights, maximumNumberOfRendererSupportedPointLights,
                    maximumNumberOfRendererSupportedSpotLights, maximumNumberOfRendererSupportedBoxLights);


                // Next is needed for forward.
#ifdef SINGLE_PASS_FORWARD_PIPELINE
                const char* shaderStr = renderer->getColoredShaderSource();

                auto deferredShader = shaderLoader.load<MVPColorShader>(shaderStr,
                                                                        {{ GL_FRAGMENT_SHADER,
                                                                           "#define DEFERRED",
                                                                           1 }});
                auto depthShader    = shaderLoader.load<MVPColorShader>(shaderStr,
                                                                     {{ GL_FRAGMENT_SHADER,
                                                                        "#define DEPTH",
                                                                        1 }});

                ShaderPart::ShaderCodeInjections sci;
                sci.emplace_back(GL_VERTEX_SHADER, "#define FORWARD_LIT", 1);
                sci.emplace_back(GL_FRAGMENT_SHADER, "#define FORWARD_LIT", 1);

                sci.emplace_back(
                    GL_FRAGMENT_SHADER,
                    "#define MAX_DL_COUNT" + std::to_string(maximumNumberOfRendererSupportedDirectionalLights), 2);
                sci.emplace_back(GL_FRAGMENT_SHADER,
                                 "#define MAX_PL_COUNT" + std::to_string(maximumNumberOfRendererSupportedPointLights),
                                 3);
                sci.emplace_back(GL_FRAGMENT_SHADER,
                                 "#define MAX_SL_COUNT" + std::to_string(maximumNumberOfRendererSupportedSpotLights),
                                 4);
                sci.emplace_back(GL_FRAGMENT_SHADER,
                                 "#define MAX_BL_COUNT" + std::to_string(maximumNumberOfRendererSupportedBoxLights), 5);
                auto forwardShader = shaderLoader.load<MVPColorShaderFL>(shaderStr, sci);

                auto wireframeShader = shaderLoader.load<MVPColorShader>(shaderStr);

                std::static_pointer_cast<ColoredAsset>(show.asset)
                    ->setShader(deferredShader, forwardShader, depthShader, wireframeShader);
#endif
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();


            int32_t count = renderer->lighting.pointLights.size();
            if (ImGui::InputInt("Point Light Count (wanted)", &count))
            {
                if (count > maximumNumberOfRendererSupportedPointLights)
                    count = maximumNumberOfRendererSupportedPointLights;
                if (count > renderer->lighting.pointLights.size())
                {
                    count -= renderer->lighting.pointLights.size();
                    for (int32_t i = 0; i < count; ++i)
                    {
                        std::shared_ptr<PointLight> light = std::make_shared<PointLight>();
                        light->setAttenuation(AttenuationPresets::Quadratic);
                        light->setIntensity(1);


                        light->setRadius(linearRand(0.1f, 4.0f));

                        light->setPosition(linearRand(vec3(-16, 0.05f, -16), vec3(16, light->getRadius(), 16)));

                        light->setColorDiffuse(linearRand(vec3(0, 0, 0), vec3(1, 1, 1)));
                        light->calculateModel();

                        light->disableShadows();
                        renderer->lighting.AddLight(light);
                    }
                }
                else if (count < renderer->lighting.pointLights.size())
                {
                    count = renderer->lighting.pointLights.size() - count;
                    for (int32_t i = 0; i < count; ++i)
                    {
                        renderer->lighting.pointLights.erase(--renderer->lighting.pointLights.end());
                    }
                }
            }
            if (ImGui::Button("Normalize Point Lights"))
            {
                for (auto pl : renderer->lighting.pointLights)
                {
                    float intensity = 1.0f / pl->getRadius();
                    pl->setIntensity(intensity);
                }
            }

            count = renderer->lighting.spotLights.size();
            if (ImGui::InputInt("Spot Light Count (wanted)", &count))
            {
                if (count > maximumNumberOfRendererSupportedSpotLights)
                    count = maximumNumberOfRendererSupportedSpotLights;
                if (count > renderer->lighting.spotLights.size())
                {
                    count -= renderer->lighting.spotLights.size();
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

                        light->disableShadows();
                        renderer->lighting.AddLight(light);
                    }
                }
                else if (count < renderer->lighting.spotLights.size())
                {
                    count = renderer->lighting.spotLights.size() - count;
                    for (int32_t i = 0; i < count; ++i)
                    {
                        renderer->lighting.spotLights.erase(--renderer->lighting.spotLights.end());
                    }
                }
            }
            if (ImGui::Button("Normalize Spot Lights"))
            {
                for (auto sl : renderer->lighting.spotLights)
                {
                    float intensity = 1.0f / sl->getRadius();
                    sl->setIntensity(intensity);
                }
            }

            count = renderer->lighting.boxLights.size();
            if (ImGui::InputInt("Box Light Count (wanted)", &count))
            {
                if (count > maximumNumberOfRendererSupportedBoxLights)
                    count = maximumNumberOfRendererSupportedBoxLights;
                if (count > renderer->lighting.boxLights.size())
                {
                    count -= renderer->lighting.boxLights.size();
                    for (int32_t i = 0; i < count; ++i)
                    {
                        std::shared_ptr<BoxLight> light = std::make_shared<BoxLight>();
                        light->setIntensity(1);

                        light->setView(linearRand(vec3(-16, 0, -16), vec3(16, 1, 16)),
                                       linearRand(vec3(-16, -2, -16), vec3(16, 0, 16)), vec3(0, 1, 0));

                        light->setScale(linearRand(vec3(2, 2, 2), vec3(5, 5, 5)));

                        light->setColorDiffuse(linearRand(vec3(0, 0, 0), vec3(1, 1, 1)));
                        light->calculateModel();

                        light->disableShadows();
                        renderer->lighting.AddLight(light);
                    }
                }
                else if (count < renderer->lighting.boxLights.size())
                {
                    count = renderer->lighting.boxLights.size() - count;
                    for (int32_t i = 0; i < count; ++i)
                    {
                        renderer->lighting.boxLights.erase(--renderer->lighting.boxLights.end());
                    }
                }
            }
            if (ImGui::Button("Normalize Box Lights"))
            {
                for (auto bl : renderer->lighting.boxLights)
                {
                    float maxSize   = bl->getScale().maxCoeff();
                    float intensity = 1.0f / maxSize;
                    bl->setIntensity(intensity);
                }
            }

            count = renderer->lighting.directionalLights.size();
            if (ImGui::InputInt("Directional Light Count (wanted)", &count))
            {
                if (count > maximumNumberOfRendererSupportedDirectionalLights)
                    count = maximumNumberOfRendererSupportedDirectionalLights;
                if (count > renderer->lighting.directionalLights.size())
                {
                    count -= renderer->lighting.directionalLights.size();
                    for (int32_t i = 0; i < count; ++i)
                    {
                        std::shared_ptr<DirectionalLight> light = std::make_shared<DirectionalLight>();

                        light->disableShadows();

                        light->setAmbientIntensity(0.01);

                        vec3 dir = Random::sphericalRand(1).cast<float>();
                        if (dir.y() > 0) dir.y() *= -1;

                        light->setIntensity(0.7);
                        light->setDirection(dir);
                        renderer->lighting.AddLight(light);
                    }
                }
                else if (count < renderer->lighting.directionalLights.size())
                {
                    count = renderer->lighting.directionalLights.size() - count;
                    for (int32_t i = 0; i < count; ++i)
                    {
                        renderer->lighting.directionalLights.erase(--renderer->lighting.directionalLights.end());
                    }
                }
            }
            if (ImGui::Button("Normalize Directional Lights"))
            {
                for (auto dl : renderer->lighting.directionalLights)
                {
                    float intensity = dl->getIntensity();
                    intensity       = 1.0f / renderer->lighting.directionalLights.size();
                    dl->setIntensity(intensity);
                }
            }
            ImGui::End();
        }
    }

   private:
    SimpleAssetObject show;
};

int main(const int argc, const char* argv[])
{
    initSaigaSample();
    Sample example;
    example.run();
    return 0;
}
