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

        sponzaAsset = std::make_shared<TexturedAsset>(UnifiedModel("models/sponza/Sponza.obj"));

        sponza.asset = sponzaAsset;
        sponza.setScale(make_vec3(0.025f));
        sponza.calculateModel();

        int maxSize = ShaderStorageBuffer::getMaxShaderStorageBlockSize();

        int maximumNumberOfRendererSupportedDirectionalLights = maxSize / (int)sizeof(DirectionalLight::ShaderData);
        int maximumNumberOfRendererSupportedPointLights       = maxSize / (int)sizeof(PointLight::ShaderData);
        int maximumNumberOfRendererSupportedSpotLights        = maxSize / (int)sizeof(SpotLight::ShaderData);

        renderer->setLightMaxima(maximumNumberOfRendererSupportedDirectionalLights,
                                 maximumNumberOfRendererSupportedPointLights,
                                 maximumNumberOfRendererSupportedSpotLights);

#ifdef SINGLE_PASS_FORWARD_PIPELINE
        const char* shaderStr    = renderer->getColoredShaderSource();
        const char* shaderStrTex = renderer->getTexturedShaderSource();

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
        boxAsset->setShader(deferredShader, forwardShader, depthShader, wireframeShader);

        deferredShader = shaderLoader.load<MVPColorShader>(shaderStrTex,
                                                                {{ GL_FRAGMENT_SHADER,
                                                                   "#define DEFERRED",
                                                                   1 }});
        depthShader = shaderLoader.load<MVPColorShader>(shaderStrTex, {{ GL_FRAGMENT_SHADER, "#define DEPTH", 1 }});

        ShaderPart::ShaderCodeInjections sci;
        sci.emplace_back(GL_VERTEX_SHADER, "#define FORWARD_LIT", 1);
        sci.emplace_back(GL_FRAGMENT_SHADER, "#define FORWARD_LIT", 1);

        sci.emplace_back(GL_FRAGMENT_SHADER,
                         "#define MAX_DL_COUNT" + std::to_string(maximumNumberOfRendererSupportedDirectionalLights), 2);
        sci.emplace_back(GL_FRAGMENT_SHADER,
                         "#define MAX_PL_COUNT" + std::to_string(maximumNumberOfRendererSupportedPointLights), 3);
        sci.emplace_back(GL_FRAGMENT_SHADER,
                         "#define MAX_SL_COUNT" + std::to_string(maximumNumberOfRendererSupportedSpotLights), 4);

        forwardShader = shaderLoader.load<MVPColorShaderFL>(shaderStrTex, sci);

        wireframeShader = shaderLoader.load<MVPColorShader>(shaderStrTex);

        sponzaAsset->setShader(deferredShader, forwardShader, depthShader, wireframeShader);
#endif

        setupBenchmark();

        std::cout << "Program Initialized!" << std::endl;
    }


    void update(float dt) override
    {
        Base::update(dt);
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

    void setupBenchmark()
    {
        renderer->lighting.pointLights.clear();
        renderer->lighting.spotLights.clear();
        renderer->lighting.directionalLights.clear();
        pointLights.clear();

        Random::setSeed(SEED);
        for (int i = 0; i < lightCount; ++i)
        {
            float r     = linearRand(0.5f, 20.0f);
            float theta = (float)i / lightCount * two_pi<float>();
            extras.push_back(vec2(r, theta));
            vec2 point((r + 14.f) * cos(theta), r * sin(theta));

            auto light = std::make_shared<PointLight>();
            light->setIntensity(1);
            light->setRadius(lightSize);
            float h = linearRand(0.25f, 25.0f);
            light->setPosition(vec3(point.x(), h, point.y()));

            light->setColorDiffuse(linearRand(vec3(0, 0, 0), vec3(1, 1, 1)));

            renderer->lighting.AddLight(light);
            pointLights.push_back(light);
        }

        camera.position = vec4(-35.3095, 16.6492, 2.20442, 1);
        camera.rot      = quat(0.732921, -0.0343305, -0.678689, -0.0318128);
        camera.calculateModel();
        camera.updateFromModel();
    }

    void renderBenchmark(Camera* camera, RenderPass render_pass)
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

    void render(Camera* camera, RenderPass render_pass) override
    {
        Base::render(camera, render_pass);

        renderBenchmark(camera, render_pass);

        if (render_pass == RenderPass::GUI)
        {
            ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Once);
            ImGui::SetNextWindowSize(ImVec2(400, 200), ImGuiCond_Once);
            ImGui::Begin("Cluster Benchmark Sample");

            if (ImGui::Button("Double Lights"))
            {
                lightCount *= 2;
                setupBenchmark();
            }
            if (ImGui::Button("Half Lights"))
            {
                lightCount /= 2;
                setupBenchmark();
            }

            ImGui::Text("Lights: %d", lightCount);

            if (ImGui::SliderFloat("Light Size", &lightSize, 0.1f, 4.0f))
            {
                setupBenchmark();
            }

            ImGui::End();
        }
    }

   private:
    std::shared_ptr<TexturedAsset> sponzaAsset;
    std::vector<vec2> extras;
    int lightCount  = 128;
    float lightSize = 1;

    SimpleAssetObject sponza;

    std::vector<std::shared_ptr<PointLight>> pointLights;


    // Order
    // each 128, 256, 512, 1024, 4096, 8192, 16384
    // six plane == SP
    // cpu plane == CP
    // gpu assignment == GA

    // Deferred Light Volumes

    // Forward
    // Forward Depth Prepass
    // Forward Tiled 64x64 - SP, CP
    // Forward Depth Prepass Tiled 64x64 - SP, CP
    // Forward Clustered 64x64, 24 depth splits - SP, CP, GA
    // Forward Depth Prepass Clustered 64x64, 24 depth splits - SP, CP, GA

    // Deferred
    // Deferred Tiled 64x64 - SP, CP
    // Deferred Clustered 64x64, 24 depth splits - SP, CP, GA
};

int main(const int argc, const char* argv[])
{
    initSaigaSample();
    Sample example;
    example.run();
    return 0;
}
