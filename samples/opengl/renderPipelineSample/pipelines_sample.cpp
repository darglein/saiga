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

#define LIGHT_SEED 9

class Sample : public RendererSampleWindow
{
    using Base                                            = RendererSampleWindow;
    int maximumNumberOfRendererSupportedDirectionalLights = 256;
    int maximumNumberOfRendererSupportedPointLights       = 256;
    int maximumNumberOfRendererSupportedSpotLights        = 256;

   public:
    Sample()
    {
        Random::setSeed(LIGHT_SEED);  // SEED
        ObjAssetLoader assetLoader;
        auto showAsset = assetLoader.loadDebugPlaneAsset(vec2(20, 20));
        // auto showAsset = assetLoader.loadColoredAsset("show_model.obj");

        show.asset = showAsset;
        show.setPosition(vec4(0.0, -0.1, 0.0, 0.0));
        // show.multScale(make_vec3(0.01f));
        show.calculateModel();

        // FIXME Remove:

        camera.position = vec4(0.303574, 2.61311, -1.76473, 1);
        camera.rot      = quat(0.953246, -0.302067, -0.00831885, -0.00263611);
        Random::setSeed(LIGHT_SEED);

        std::shared_ptr<PointLight> light = std::make_shared<PointLight>();
        light->setRadius(linearRand(0.5, 4.0));
        light->setIntensity(1.0f / light->getRadius());
        light->setPosition(linearRand(vec3(-16, light->getRadius() * 0.5, -16), vec3(16, light->getRadius(), 16)));
        light->setColorDiffuse(linearRand(vec3(0, 0, 0), vec3(1, 1, 1)));
        light->castShadows = false;
        renderer->lighting.AddLight(light);
        pointLights.push_back(light);

        // End

        int maxSize = ShaderStorageBuffer::getMaxShaderStorageBlockSize();

        maximumNumberOfRendererSupportedDirectionalLights = std::clamp(
            maximumNumberOfRendererSupportedDirectionalLights, 0, maxSize / (int)sizeof(DirectionalLight::ShaderData));
        maximumNumberOfRendererSupportedPointLights =
            std::clamp(maximumNumberOfRendererSupportedPointLights, 0, maxSize / (int)sizeof(PointLight::ShaderData));
        maximumNumberOfRendererSupportedSpotLights =
            std::clamp(maximumNumberOfRendererSupportedSpotLights, 0, maxSize / (int)sizeof(SpotLight::ShaderData));

// Next is needed for forward.
#ifdef SINGLE_PASS_FORWARD_PIPELINE
        const char* shaderStr = renderer->getColoredShaderSource();

        renderer->setLightMaxima(maximumNumberOfRendererSupportedDirectionalLights,
                                 maximumNumberOfRendererSupportedPointLights,
                                 maximumNumberOfRendererSupportedSpotLights);

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
            supportChanged |= ImGui::InputInt("Renderer Supported Directional Lights",
                                              &maximumNumberOfRendererSupportedDirectionalLights);


            if (supportChanged)
            {
                int maxSize = ShaderStorageBuffer::getMaxShaderStorageBlockSize();

                maximumNumberOfRendererSupportedDirectionalLights =
                    std::clamp(maximumNumberOfRendererSupportedDirectionalLights, 0,
                               maxSize / (int)sizeof(DirectionalLight::ShaderData));
                maximumNumberOfRendererSupportedPointLights = std::clamp(maximumNumberOfRendererSupportedPointLights, 0,
                                                                         maxSize / (int)sizeof(PointLight::ShaderData));
                maximumNumberOfRendererSupportedSpotLights  = std::clamp(maximumNumberOfRendererSupportedSpotLights, 0,
                                                                        maxSize / (int)sizeof(SpotLight::ShaderData));

                renderer->setLightMaxima(maximumNumberOfRendererSupportedDirectionalLights,
                                         maximumNumberOfRendererSupportedPointLights,
                                         maximumNumberOfRendererSupportedSpotLights);


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
                renderer->lighting.pointLights.clear();
                pointLights.clear();
                Random::setSeed(LIGHT_SEED);
                for (int32_t i = 0; i < count; ++i)
                {
                    std::shared_ptr<PointLight> light = std::make_shared<PointLight>();
                    light->setRadius(linearRand(0.5, 4.0));
                    light->setIntensity(1.0f / light->getRadius());
                    light->setPosition(
                        linearRand(vec3(-16, light->getRadius() * 0.5, -16), vec3(16, light->getRadius(), 16)));
                    light->setColorDiffuse(linearRand(vec3(0, 0, 0), vec3(1, 1, 1)));
                    light->castShadows = false;
                    renderer->lighting.AddLight(light);
                    pointLights.push_back(light);
                }
            }

            count = renderer->lighting.spotLights.size();
            if (ImGui::InputInt("Spot Light Count (wanted)", &count))
            {
                if (count > maximumNumberOfRendererSupportedSpotLights)
                    count = maximumNumberOfRendererSupportedSpotLights;
                renderer->lighting.spotLights.clear();
                spotLights.clear();
                Random::setSeed(LIGHT_SEED);
                for (int32_t i = 0; i < count; ++i)
                {
                    std::shared_ptr<SpotLight> light = std::make_shared<SpotLight>();
                    light->setRadius(linearRand(1.0, 4.0));
                    light->setIntensity(1.0);
                    light->setPosition(linearRand(vec3(-16, 1, -16), vec3(16, light->getRadius(), 16)));
                    light->setAngle(linearRand(25, 55));
                    light->setColorDiffuse(linearRand(vec3(0, 0, 0), vec3(1, 1, 1)));
                    light->direction   = linearRand(vec3(-0.5, -0.5, -0.5), vec3(0.5, 0.5, 0.5));
                    light->castShadows = false;
                    renderer->lighting.AddLight(light);
                    spotLights.push_back(light);
                }
            }

            count = renderer->lighting.directionalLights.size();
            if (ImGui::InputInt("Directional Light Count (wanted)", &count))
            {
                if (count > maximumNumberOfRendererSupportedDirectionalLights)
                    count = maximumNumberOfRendererSupportedDirectionalLights;
                renderer->lighting.directionalLights.clear();
                directionalLights.clear();
                Random::setSeed(LIGHT_SEED);
                for (int32_t i = 0; i < count; ++i)
                {
                    std::shared_ptr<DirectionalLight> light = std::make_shared<DirectionalLight>();
                    light->createShadowMap(1, 1, 1, ShadowQuality::LOW);  // FIXME Has to be called?
                    light->castShadows = false;
                    light->setAmbientIntensity(0.01);
                    vec3 dir = Random::sphericalRand(1).cast<float>();
                    if (dir.y() > 0) dir.y() *= -1;
                    light->setIntensity(0.75);
                    light->setDirection(dir);
                    renderer->lighting.AddLight(light);
                    directionalLights.push_back(light);
                }
            }

            ImGui::End();
        }
    }

   private:
    SimpleAssetObject show;

    std::vector<std::shared_ptr<PointLight>> pointLights;
    std::vector<std::shared_ptr<SpotLight>> spotLights;
    std::vector<std::shared_ptr<DirectionalLight>> directionalLights;
};

int main(const int argc, const char* argv[])
{
    initSaigaSample();
    Sample example;
    example.run();
    return 0;
}
