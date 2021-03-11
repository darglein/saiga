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

#define COLOR_SEED 9

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

        setupPlayground(currentPlayground);

        std::cout << "Program Initialized!" << std::endl;
    }


    // void update(float dt) override { Base::update(dt); }

    void setupPlayground(int index)
    {
        int maxSize = ShaderStorageBuffer::getMaxShaderStorageBlockSize();

        int maximumNumberOfRendererSupportedDirectionalLights = maxSize / (int)sizeof(DirectionalLight::ShaderData);
        int maximumNumberOfRendererSupportedPointLights       = maxSize / (int)sizeof(PointLight::ShaderData);
        int maximumNumberOfRendererSupportedSpotLights        = maxSize / (int)sizeof(SpotLight::ShaderData);

        renderer->setLightMaxima(maximumNumberOfRendererSupportedDirectionalLights,
                                 maximumNumberOfRendererSupportedPointLights,
                                 maximumNumberOfRendererSupportedSpotLights);
        switch (index)
        {
            case 0:
            {
                Random::setSeed(COLOR_SEED);
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
                dl->setIntensity(0.1f);
                renderer->lighting.AddLight(dl);

                plane.asset = std::make_shared<ColoredAsset>(PlaneMesh(Plane(vec3(0, -0.1, 0), vec3(0, 1, 0))));

                plane.setScale(vec3(20, 1, 20));
                plane.calculateModel();

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

                static_cast<ColoredAsset*>(plane.asset.get())
                    ->setShader(deferredShader, forwardShader, depthShader, wireframeShader);
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

            ImGui::Combo("PlaygroundScene", &currentPlayground, descriptions, sceneCount);

            ImGui::End();
        }
    }

   private:
    SimpleAssetObject plane;

    std::vector<std::shared_ptr<PointLight>> pointLights;
    std::vector<std::shared_ptr<SpotLight>> spotLights;

    int currentPlayground = 0;

    static const int sceneCount = 6;

    const char* descriptions[sceneCount] = {"SIMPLE_PLANE",         "SPONZA",         "COMPLEX_DEPTH", "BLOCKED_VIEW",
                                            "PERFECT_DISTRIBUTION", "ALL_IN_ONE_SPOT"};
};

int main(const int argc, const char* argv[])
{
    initSaigaSample();
    Sample example;
    example.run();
    return 0;
}
