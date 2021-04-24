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
        // Define GUI layout
        auto editor_layout = std::make_unique<EditorLayoutL>();
        editor_layout->RegisterImguiWindow("Rendering Lighting Sample", EditorLayoutL::WINDOW_POSITION_LEFT_BOTTOM);
        editor_gui.SetLayout(std::move(editor_layout));

        Random::setSeed(LIGHT_SEED);  // SEED

        // show.asset = std::make_shared<ColoredAsset>(
        //    CheckerBoardPlane(make_ivec2(40, 40), 1.0f, Colors::darkgray, Colors::white));

        show.asset = std::make_shared<TexturedAsset>(
            UnifiedModel("C:/Users/paulh/Documents/gltf_2_0_sample_models/2.0/Sponza/glTF/Sponza.gltf").Normalize());

        // show.setPosition(vec4(0.0, -0.1, 0.0, 0.0));

        // test
        float aspect = window->getAspectRatio();
        camera.setProj(60.0f, aspect, 0.1f, 5.0f);
        camera.setView(vec3(0, 1, 2), vec3(0, 0, 0), vec3(0, 1, 0));
        camera.movementSpeed     = 0.3;
        camera.movementSpeedFast = 3;
        camera.position          = vec4(0.558927, 0.0488419, 0.00189565, 1);
        camera.rot               = quat(0.72404, -0.060576, 0.684689, 0.0572873);



        int maxSize = ShaderStorageBuffer::getMaxShaderStorageBlockSize();

        maximumNumberOfRendererSupportedDirectionalLights = std::clamp(
            maximumNumberOfRendererSupportedDirectionalLights, 0, maxSize / (int)sizeof(DirectionalLight::ShaderData));
        maximumNumberOfRendererSupportedPointLights =
            std::clamp(maximumNumberOfRendererSupportedPointLights, 0, maxSize / (int)sizeof(PointLight::ShaderData));
        maximumNumberOfRendererSupportedSpotLights =
            std::clamp(maximumNumberOfRendererSupportedSpotLights, 0, maxSize / (int)sizeof(SpotLight::ShaderData));

        renderer->setLightMaxima(maximumNumberOfRendererSupportedDirectionalLights,
                                 maximumNumberOfRendererSupportedPointLights,
                                 maximumNumberOfRendererSupportedSpotLights);

// Next is needed for forward.
#ifdef SINGLE_PASS_FORWARD_PIPELINE
        const char* shaderStr = renderer->getTexturedShaderSource();

        auto deferredShader = shaderLoader.load<MVPTextureShader>(shaderStr,
                                                                  {{ GL_FRAGMENT_SHADER,
                                                                     "#define DEFERRED",
                                                                     1 }});
        auto depthShader = shaderLoader.load<MVPTextureShader>(shaderStr, {{ GL_FRAGMENT_SHADER, "#define DEPTH", 1 }});

        ShaderPart::ShaderCodeInjections sci;
        sci.emplace_back(GL_VERTEX_SHADER, "#define FORWARD_LIT", 1);
        sci.emplace_back(GL_FRAGMENT_SHADER, "#define FORWARD_LIT", 1);

        sci.emplace_back(GL_FRAGMENT_SHADER,
                         "#define MAX_DL_COUNT" + std::to_string(maximumNumberOfRendererSupportedDirectionalLights), 2);
        sci.emplace_back(GL_FRAGMENT_SHADER,
                         "#define MAX_PL_COUNT" + std::to_string(maximumNumberOfRendererSupportedPointLights), 3);
        sci.emplace_back(GL_FRAGMENT_SHADER,
                         "#define MAX_SL_COUNT" + std::to_string(maximumNumberOfRendererSupportedSpotLights), 4);
        auto forwardShader = shaderLoader.load<MVPTextureShaderFL>(shaderStr, sci);

        auto wireframeShader = shaderLoader.load<MVPTextureShader>(shaderStr);

        std::static_pointer_cast<TexturedAsset>(show.asset)
            ->setShader(deferredShader, forwardShader, depthShader, wireframeShader);
#endif

        std::cout << "Program Initialized!" << std::endl;
    }



    void render(Camera* camera, RenderPass render_pass) override
    {
        Base::render(camera, render_pass);

        if (render_pass == RenderPass::Shadow)
        {
            show.renderDepth(camera);
        }
#if defined(SINGLE_PASS_DEFERRED_PIPELINE) || defined(MULTI_PASS_DEFERRED_PIPELINE)
        if (render_pass == RenderPass::Deferred)
        {
            show.render(camera);
        }
#elif defined(SINGLE_PASS_FORWARD_PIPELINE)
        if (render_pass == RenderPass::DepthPrepass)
        {
            show.renderDepth(camera);
        }
        if (render_pass == RenderPass::Forward)
        {
            show.renderForward(camera);
        }
#endif
        if (render_pass == RenderPass::GUI)
        {
            ImGui::Begin("Rendering Lighting Sample");

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
                const char* shaderStr = renderer->getTexturedShaderSource();

                auto deferredShader = shaderLoader.load<MVPTextureShader>(shaderStr,
                                                                          {{ GL_FRAGMENT_SHADER,
                                                                             "#define DEFERRED",
                                                                             1 }});
                auto depthShader    = shaderLoader.load<MVPTextureShader>(shaderStr,
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
                auto forwardShader = shaderLoader.load<MVPTextureShaderFL>(shaderStr, sci);

                auto wireframeShader = shaderLoader.load<MVPTextureShader>(shaderStr);

                std::static_pointer_cast<TexturedAsset>(show.asset)
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
                        linearRand(vec3(-36, light->getRadius() * 0.5, -36), vec3(36, light->getRadius(), 36)));
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
                    light->setIntensity(1.0f / light->getRadius());
                    light->setPosition(
                        linearRand(vec3(-36, light->getRadius() * 0.5, -36), vec3(36, light->getRadius(), 36)));
                    light->setColorDiffuse(linearRand(vec3(0, 0, 0), vec3(1, 1, 1)));
                    light->setAngle(linearRand(20, 60));
                    light->direction   = linearRand(vec3(-0.5, -1, -0.5), vec3(0.5, -1, 0.5));
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
