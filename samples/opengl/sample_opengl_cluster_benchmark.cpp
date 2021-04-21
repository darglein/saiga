/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/math/random.h"
#include "saiga/core/model/model_from_shape.h"
#include "saiga/core/util/statistics.h"
#include "saiga/opengl/rendering/lighting/cpu_plane_clusterer.h"
#include "saiga/opengl/rendering/lighting/gpu_assignment_clusterer.h"
#include "saiga/opengl/rendering/lighting/renderer_lighting.h"
#include "saiga/opengl/rendering/lighting/six_plane_clusterer.h"
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
        editor_layout->RegisterImguiWindow("Cluster Benchmark Sample", EditorLayoutL::WINDOW_POSITION_LEFT);
        editor_gui.SetLayout(std::move(editor_layout));

        sponzaAsset = std::make_shared<ColoredAsset>(
            UnifiedModel("C:/Users/paulh/Documents/gltf_2_0_sample_models/2.0/Sponza/glTF/Sponza.gltf"));
        // sponzaAsset =
        // std::make_shared<ColoredAsset>(UnifiedModel("C:/Users/paulh/Documents/gltf_2_0_sample_models/lumberyard_bistro/BistroExterior.gltf"));

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
        const char* shaderStrTex = renderer->getColoredShaderSource();
        auto deferredShader      = shaderLoader.load<MVPColorShader>(shaderStrTex,
                                                                {{ GL_FRAGMENT_SHADER,
                                                                   "#define DEFERRED",
                                                                   1 }});
        auto depthShader         = shaderLoader.load<MVPColorShader>(shaderStrTex,
                                                             {{ GL_FRAGMENT_SHADER,
                                                                "#define DEPTH",
                                                                1 }});

        ShaderPart::ShaderCodeInjections sci;
        sci.emplace_back(GL_VERTEX_SHADER, "#define FORWARD_LIT", 1);
        sci.emplace_back(GL_FRAGMENT_SHADER, "#define FORWARD_LIT", 1);

        sci.emplace_back(GL_FRAGMENT_SHADER,
                         "#define MAX_DL_COUNT" + std::to_string(maximumNumberOfRendererSupportedDirectionalLights), 2);
        sci.emplace_back(GL_FRAGMENT_SHADER,
                         "#define MAX_PL_COUNT" + std::to_string(maximumNumberOfRendererSupportedPointLights), 3);
        sci.emplace_back(GL_FRAGMENT_SHADER,
                         "#define MAX_SL_COUNT" + std::to_string(maximumNumberOfRendererSupportedSpotLights), 4);

        auto forwardShader = shaderLoader.load<MVPColorShaderFL>(shaderStrTex, sci);

        auto wireframeShader = shaderLoader.load<MVPColorShader>(shaderStrTex);

        sponzaAsset->setShader(deferredShader, forwardShader, depthShader, wireframeShader);
#endif

        lightCount = 128;
        lightSize  = 1.0f;
        setupBenchmark();

        std::cout << "Program Initialized!" << std::endl;

        int w = 1920;
        int h = 1080;

        std::unique_ptr<Framebuffer> target_framebuffer;
        target_framebuffer = std::make_unique<Framebuffer>();
        target_framebuffer->create();

        std::shared_ptr<Texture> color = std::make_shared<Texture>();
        color->create(w, h, GL_RGBA, GL_RGBA8, GL_UNSIGNED_BYTE);

        std::shared_ptr<Texture> depth_stencil = std::make_shared<Texture>();
        depth_stencil->create(w, h, GL_DEPTH_STENCIL, GL_DEPTH24_STENCIL8, GL_UNSIGNED_INT_24_8);

        target_framebuffer->attachTexture(color);
        target_framebuffer->attachTextureDepthStencil(depth_stencil);
        target_framebuffer->check();

        ViewPort vp;
        vp.position = ivec2(0, 0);
        vp.size     = ivec2(w, h);

        int rendererTypesToCheck = 9;
        int settingsToCheck      = 4;
#ifdef MULTI_PASS_DEFERRED_PIPELINE
        rendererTypesToCheck = 1;
        settingsToCheck      = 1;
#endif

        std::string rendererTypes[9] = {"BASIC",  "TLD SP", "TLD CP",     "TLD CP REF", "TLD GA",
                                        "CLD SP", "CLD CP", "CLD CP REF", "CLD GA"};

        int lightCounts[8] = {256, 512, 1024, 2048, 4096, 8192, 16384, 32768};

        std::vector<std::pair<std::string, std::vector<double>>> timeColumns;

        for (int r = 0; r < rendererTypesToCheck; ++r)
        {
            setupRenderer(r);
            int _settingsToCheck = r > 0 ? settingsToCheck : 1;
            for (int s = 0; s < _settingsToCheck; ++s)
            {
                setupRendererSettings(r, s);
                std::string name = rendererTypes[r] + std::to_string(s);
                timeColumns.push_back({name, {}});
                auto& medians = timeColumns.at(timeColumns.size() - 1).second;
                Statistics stats;
                for (int l = 0; l < 7; ++l)
                {
                    lightCount = lightCounts[l];
                    if (stats.median > 50.0f) break;
                    setupBenchmark();
                    // Discard first frames.
                    for (int i = 0; i < 4; ++i)
                    {
                        renderer->renderGL(target_framebuffer.get(), vp, &camera);
                    }

                    std::vector<double> times;
                    for (int i = 0; i < 60; ++i)
                    {
                        OpenGLTimer tim;
                        tim.start();
                        renderer->renderGL(target_framebuffer.get(), vp, &camera);
                        tim.stop();
                        times.push_back(tim.getTimeMS());
                    }
                    stats = Statistics(times);
                    medians.push_back(stats.median);
                }
            }
        }


        TemplatedImage<ucvec4> result(h, w);
        target_framebuffer->getTextureColor(0)->download(result.data());

        result.save("output.png");

#ifdef SINGLE_PASS_DEFERRED_PIPELINE
        std::ofstream timesOut("deferred_times.csv");
#elif defined(MULTI_PASS_DEFERRED_PIPELINE)
        std::ofstream timesOut("light_volume_times.csv");
#elif defined(SINGLE_PASS_FORWARD_PIPELINE)
        std::ofstream timesOut("forward_times.csv");
#endif
        timesOut << "LightCount";
        timesOut << ";";
        for (int i = 0; i < timeColumns.size(); ++i)
        {
            timesOut << timeColumns.at(i).first;
            if (i != timeColumns.size() - 1) timesOut << ";";
        }
        timesOut << "\n";
        for (int i = 0; i < 7; ++i)
        {
            timesOut << lightCounts[i];
            timesOut << ";";
            for (int j = 0; j < timeColumns.size(); ++j)
            {
                if (i < timeColumns.at(j).second.size()) timesOut << timeColumns.at(j).second.at(i);
                if (j != timeColumns.size() - 1) timesOut << ";";
            }
            timesOut << "\n";
        }

        timesOut.close();

        exit(0);
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
            light->setIntensity(2);
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

    void setupRenderer(int rendererIndex)
    {
        switch (rendererIndex)
        {
            case 0:
                // BASIC
                renderer->lighting.setClusterType(0);
                break;
            case 1:
                // TLD SP
                renderer->lighting.setClusterType(1);
                break;
            case 2:
                // TLD CP
                renderer->lighting.setClusterType(2);
                break;
            case 3:
                // TLD CP REF
                renderer->lighting.setClusterType(2);
                break;
            case 4:
                // TLD GA
                renderer->lighting.setClusterType(3);
                break;
            case 5:
                // CLD SP
                renderer->lighting.setClusterType(1);
                break;
            case 6:
                // CLD CP
                renderer->lighting.setClusterType(2);
                break;
            case 7:
                // CLD CP REF
                renderer->lighting.setClusterType(2);
                break;
            case 8:
                // CLD GA
                renderer->lighting.setClusterType(3);
                break;
            default:
                break;
        }
    }

    void setupRendererSettings(int rendererIndex, int settingsIndex)
    {
        static int tiledTileSettings[4]     = {32, 64, 128, 256};
        static int clusteredTileSettings[4] = {32, 64, 64, 256};
        static int depthSplitSettings[4]    = {6, 16, 24, 64};
        switch (rendererIndex)
        {
            case 0:
                // BASIC
                break;
            case 2:
                // TLD CP
                std::static_pointer_cast<CPUPlaneClusterer>(renderer->lighting.getClusterer())->refinement = false;
            case 3:
                // TLD CP REF
                std::static_pointer_cast<CPUPlaneClusterer>(renderer->lighting.getClusterer())->refinement = true;
            case 1:
                // TLD SP
            case 4:
                // TLD GA
                renderer->lighting.getClusterer()->set(tiledTileSettings[settingsIndex], 0);
                break;
            case 6:
                // CLD CP
                std::static_pointer_cast<CPUPlaneClusterer>(renderer->lighting.getClusterer())->refinement = false;
            case 7:
                // CLD CP REF
                std::static_pointer_cast<CPUPlaneClusterer>(renderer->lighting.getClusterer())->refinement = true;
            case 5:
                // CLD SP
            case 8:
                // CLD GA
                renderer->lighting.getClusterer()->set(clusteredTileSettings[settingsIndex],
                                                       depthSplitSettings[settingsIndex]);
                break;
            default:
                break;
        }
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
    std::shared_ptr<ColoredAsset> sponzaAsset;
    std::vector<vec2> extras;
    int lightCount  = 128;
    float lightSize = 1;

    SimpleAssetObject sponza;

    std::vector<std::shared_ptr<PointLight>> pointLights;
};

int main(const int argc, const char* argv[])
{
    initSaigaSample();
    Sample example;
    example.run();
    return 0;
}
