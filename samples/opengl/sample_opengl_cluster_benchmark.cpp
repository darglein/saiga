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

#define SPONZA
//#define SPONZA_LARGE
//#define BISTRO
//#define BISTRO_LINE
//#define BISTRO_POINT

#define MEDIAN

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

#ifdef SPONZA
        TheAsset = std::make_shared<ColoredAsset>(
            UnifiedModel("D:/Users/paulh/Documents/gltf_2_0_sample_models/2.0/Sponza/glTF/Sponza.gltf"));

        assetObject.asset = TheAsset;
        assetObject.setScale(make_vec3(0.025f));
        assetObject.calculateModel();
#else
        auto model =
            UnifiedModel("D:/Users/paulh/Documents/gltf_2_0_sample_models/lumberyard_bistro/BistroExterior.gltf");
        model    = model.Normalize();
        TheAsset = std::make_shared<ColoredAsset>(model);
        bb       = model.CombinedMesh().first.BoundingBox();
        bb.scale(make_vec3(50.0));
        assetObject.asset = TheAsset;
        assetObject.setScale(make_vec3(50.0));
        assetObject.calculateModel();
#endif

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

        TheAsset->setShader(deferredShader, forwardShader, depthShader, wireframeShader);
#endif

        lightCount = 128;
#ifdef SPONZA_LARGE
        lightSize = 6.0f;
#else
        lightSize = 2.0f;
#endif

        Random::setSeed(SEED);
        int maxLights = 16384;
#ifdef SPONZA
        start_position  = vec4(28.2392, 17.1928, 11.2081, 1);
        start_rot       = quat(0.88629, -0.0913254, 0.451591, 0.0465218);
        middle_position = vec4(-29.1124, 17.4062, 11.2679, 1);
        middle_rot      = quat(0.885287, -0.100587, -0.45107, -0.0512778);
        goal_position   = vec4(-28.6951, 14.2356, -3.93452, 1);
        goal_rot        = quat(0.642224, -0.0491997, -0.762658, -0.0584746);

        for (int i = 0; i < maxLights; ++i)
        {
            float r     = linearRand(0.5f, 20.0f);
            float theta = linearRand(0, maxLights) / two_pi<float>();
            extras.push_back(vec2(r, theta));
            vec2 point((r + 14.f) * cos(theta), r * sin(theta));

            auto light = std::make_shared<PointLight>();
            light->setIntensity(2);
            light->setRadius(lightSize);
            float h = linearRand(0.25f, 25.0f);
            light->setPosition(vec3(point.x(), h, point.y()));

            light->setColorDiffuse(linearRand(vec3(0, 0, 0), vec3(1, 1, 1)));

            pointLights.push_back(light);
        }
#elif defined(BISTRO)
        start_position = vec4(-13.9085, -7.96026, 2.92398, 1);
        start_rot      = quat(0.981276, -0.0356852, 0.188964, 0.00688365);
        goal_position  = vec4(-14.353, -8.72036, -0.694178, 1);
        goal_rot       = quat(0.880712, 0.0495057, 0.470257, -0.0264335);

        camera.position = goal_position;
        camera.rot      = goal_rot;
        camera.calculateModel();
        camera.updateFromModel();

        for (int i = 0; i < maxLights; ++i)
        {
            float z   = -linearRand(camera.zNear + 1, camera.zFar * 0.75);
            float d   = -((z - camera.zNear) / (camera.zFar - camera.zNear));
            auto fovx = degrees(camera.fovy) * camera.aspect;
            float r   = linearRand(-d * fovx, d * fovx);
            float h   = linearRand(-0.035 * degrees(camera.fovy), 0.35 * degrees(camera.fovy));
            vec3 p    = make_vec3(camera.position);
#    if not defined(BISTRO_LINE) or defined(BISTRO_POINT)
            p += make_vec3(camera.getDirection().array() * z);
            p += make_vec3(camera.getRightVector().array() * r);
            p += make_vec3(camera.getUpVector().array() * h);
#    endif

#    ifdef BISTRO_LINE
            p += make_vec3(camera.getDirection().array() * z);
#    endif

#    ifdef BISTRO_POINT
            p += make_vec3(camera.getDirection().array() * -28);
            p += make_vec3(camera.getUpVector().array() * -1.0);
#    endif

            auto light = std::make_shared<PointLight>();
            light->setIntensity(1);
            light->setRadius(lightSize);
            light->setPosition(vec3(p.x(), p.y(), p.z()));

            light->setColorDiffuse(linearRand(vec3(0, 0, 0), vec3(1, 1, 1)));

            pointLights.push_back(light);
        }
#endif

        setupBenchmark();

        std::cout << "Program Initialized!" << std::endl;

//#define OFFLINE_BENCHMARK
#ifdef OFFLINE_BENCHMARK
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

        int rendererTypesToCheck = 6;
#    ifdef MULTI_PASS_DEFERRED_PIPELINE
        rendererTypesToCheck = 1;
#    endif

        std::string rendererTypes[6] = {"BASIC", "TLD SP", "TLD ISR", "CLD SP", "CLD ISR", "CLD GA"};

#    ifdef MEDIAN
        int iterateCount = 126;  // 125 + 1, da wir bei 0 anfangen
#    else
        int iterateCount = 1;
#    endif

        std::vector<std::pair<std::string, std::vector<double>>> timeColumns;

        for (int r = 0; r < rendererTypesToCheck; ++r)
        {
#    ifndef MEDIAN
            r = r == 0 ? 1 : r;
#    endif
            setupRenderer(r);
            std::string name = rendererTypes[r];
            timeColumns.push_back({name, {}});
#    ifdef MEDIAN
            auto& medians = timeColumns.at(timeColumns.size() - 1).second;
#    else
            auto& frametimes = timeColumns.at(timeColumns.size() - 1).second;
#    endif
            Statistics stats;
            lightCount = 0;
            for (int l = 0; l < iterateCount; ++l)
            {
                if (stats.median > 42.0f) break;  // 42 ms -> ~ 24 fps
#    ifndef MEDIAN
                lightCount = 16000;
#    endif
                setupBenchmark();
#    ifdef MEDIAN
                lightCount += 128;
#    endif
                // Discard first frames.
                for (int i = 0; i < 4; ++i)
                {
                    renderer->renderGL(target_framebuffer.get(), vp, &camera);
                }

                std::vector<double> times;
                currentFrameUpdate = 0;
                for (int i = 0; i < measureFrameCount; ++i)
                {
                    update(0.0166667f);
                    OpenGLTimer tim;
                    tim.start();
                    renderer->renderGL(target_framebuffer.get(), vp, &camera);
                    tim.stop();
                    times.push_back(tim.getTimeMS());
#    ifndef MEDIAN
                    frametimes.push_back(tim.getTimeMS());
                    continue;
#    endif
                    // if (i == 0)
                    //{
                    //    TemplatedImage<ucvec4> result(h, w);
                    //    target_framebuffer->getTextureColor(0)->download(result.data());

                    //    result.save("frame0.png");
                    //}
                }
#    ifdef MEDIAN
                stats = Statistics(times);
                medians.push_back(stats.median);
#    endif
            }
        }


        TemplatedImage<ucvec4> result(h, w);
        target_framebuffer->getTextureColor(0)->download(result.data());

        result.save("output.png");

#    ifdef SINGLE_PASS_DEFERRED_PIPELINE
        std::ofstream timesOut("deferred_times_one_point_bistro.csv");
#    elif defined(MULTI_PASS_DEFERRED_PIPELINE)
        std::ofstream timesOut("light_volume_times_one_point_bistro.csv");
#    elif defined(SINGLE_PASS_FORWARD_PIPELINE)
        std::ofstream timesOut("forward_times_bistro.csv");
#    endif
        timesOut << "LightCount";
        timesOut << ";";
        for (int i = 0; i < timeColumns.size(); ++i)
        {
            timesOut << timeColumns.at(i).first;
            if (i != timeColumns.size() - 1) timesOut << ";";
        }
        timesOut << "\n";
#    ifdef MEDIAN
        lightCount = 0;
        for (int i = 0; i < iterateCount; ++i)
        {
            timesOut << lightCount;
            timesOut << ";";
            lightCount += 128;
            for (int j = 0; j < timeColumns.size(); ++j)
            {
                if (i < timeColumns.at(j).second.size()) timesOut << timeColumns.at(j).second.at(i);
                if (j != timeColumns.size() - 1) timesOut << ";";
            }
            timesOut << "\n";
        }
#    else
        for (int i = 0; i < measureFrameCount; ++i)
        {
            timesOut << i;
            timesOut << ";";
            for (int j = 0; j < timeColumns.size(); ++j)
            {
                if (i < timeColumns.at(j).second.size()) timesOut << timeColumns.at(j).second.at(i);
                if (j != timeColumns.size() - 1) timesOut << ";";
            }
            timesOut << "\n";
        }
#    endif

        timesOut.close();

        exit(0);
#endif  // OFFLINE_BENCHMARK
    }


    void update(float dt) override
    {
        Base::update(dt);
#ifdef SPONZA
        for (int i = 0; i < lightCount; ++i)
        {
            vec2& ex = extras[i];
            auto pl  = pointLights[i];
            float h  = pl->getPosition().y();
            ex[1] += 0.5f * dt;
            vec2 point((ex[0] + 16.f) * cos(ex[1]), ex[0] * sin(ex[1]));
            pl->setPosition(vec3(point.x(), h, point.y()));
        }
#endif
        // interpolate camera
        float t = (float)currentFrameUpdate / (float)(measureFrameCount - 1);
#ifdef SPONZA
        auto start_p = start_position;
        auto start_r = start_rot;
        auto end_p   = middle_position;
        auto end_r   = middle_rot;

        if (t > 0.5)
        {
            start_p = middle_position;
            start_r = middle_rot;
            end_p   = goal_position;
            end_r   = goal_rot;
            t -= 0.5;
        }
        t *= 2;
#else
        auto start_p = start_position;
        auto start_r = start_rot;
        auto end_p   = goal_position;
        auto end_r   = goal_rot;
#endif


        float t_ = 1 - t;


        camera.position = start_p * t_ + end_p * t;

        camera.rot.x() = t_ * start_r.x() + t * end_r.x();
        camera.rot.y() = t_ * start_r.y() + t * end_r.y();
        camera.rot.z() = t_ * start_r.z() + t * end_r.z();
        camera.rot.w() = t_ * start_r.w() + t * end_r.w();

        if (t >= 1.0)
        {
            camera.position    = goal_position;
            camera.rot         = goal_rot;
            currentFrameUpdate = 0;
        }

        camera.rot.normalize();
        camera.calculateModel();
        camera.updateFromModel();

        currentFrameUpdate++;
    }

    void setupBenchmark()
    {
        renderer->lighting.pointLights.clear();
        renderer->lighting.spotLights.clear();
        renderer->lighting.directionalLights.clear();

#ifdef SPONZA
        for (int i = 0; i < lightCount; ++i)
        {
            auto pl = pointLights[i];
            renderer->lighting.AddLight(pl);
        }

        camera.position = start_position;
        camera.rot      = start_rot;
        camera.calculateModel();
        camera.updateFromModel();
#else
        for (int i = 0; i < lightCount; ++i)
        {
            auto pl = pointLights[i];
            renderer->lighting.AddLight(pl);
        }

        camera.position = start_position;
        camera.rot      = start_rot;
        camera.calculateModel();
        camera.updateFromModel();
#endif
    }

    void setupRenderer(int rendererIndex)
    {
#ifdef SINGLE_PASS_FORWARD_PIPELINE
        static int tileSizeSettings[6] = {0, 64, 32, 256, 64, 128};
#elif defined(SINGLE_PASS_DEFERRED_PIPELINE)
        static int tileSizeSettings[6] = {0, 128, 64, 256, 256, 128};
#else
        static int tileSizeSettings[] = {0};
#endif
        static int depthSplitSettings[3] = {4, 8, 24};
        switch (rendererIndex)
        {
            case 0:
                // BASIC
                renderer->lighting.setClusterType(0);
                break;
            case 1:
                // TLD SP
                renderer->lighting.setClusterType(1);
                renderer->lighting.getClusterer()->set(tileSizeSettings[1], 0);
                renderer->lighting.getClusterer()->enable3DClusters(false);
                break;
            case 2:
                // TLD ISR
                renderer->lighting.setClusterType(2);
                std::static_pointer_cast<CPUPlaneClusterer>(renderer->lighting.getClusterer())->refinement = true;
                renderer->lighting.getClusterer()->set(tileSizeSettings[2], 0);
                renderer->lighting.getClusterer()->enable3DClusters(false);
                break;
            case 3:
                // CLD SP
                renderer->lighting.setClusterType(1);
                renderer->lighting.getClusterer()->set(tileSizeSettings[3], depthSplitSettings[0]);
                renderer->lighting.getClusterer()->enable3DClusters(true);
                break;
            case 4:
                // CLD ISR
                renderer->lighting.setClusterType(2);
                std::static_pointer_cast<CPUPlaneClusterer>(renderer->lighting.getClusterer())->refinement = true;
                renderer->lighting.getClusterer()->set(tileSizeSettings[4], depthSplitSettings[1]);
                renderer->lighting.getClusterer()->enable3DClusters(true);
                break;
            case 5:
                // CLD GA
                renderer->lighting.setClusterType(3);
                renderer->lighting.getClusterer()->set(tileSizeSettings[5], depthSplitSettings[2]);
                renderer->lighting.getClusterer()->enable3DClusters(true);
                break;
            default:
                break;
        }
    }

    void renderBenchmark(Camera* camera, RenderPass render_pass)
    {
        if (render_pass == RenderPass::Shadow)
        {
            assetObject.renderDepth(camera);
        }
#if defined(SINGLE_PASS_DEFERRED_PIPELINE) || defined(MULTI_PASS_DEFERRED_PIPELINE)
        if (render_pass == RenderPass::Deferred)
        {
            assetObject.render(camera);
        }
#elif defined(SINGLE_PASS_FORWARD_PIPELINE)
        if (render_pass == RenderPass::DepthPrepass)
        {
            assetObject.renderDepth(camera);
        }
        if (render_pass == RenderPass::Forward)
        {
            assetObject.renderForward(camera);
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

            // if (ImGui::SliderFloat("Light Size", &lightSize, 0.1f, 4.0f))
            //{
            //    setupBenchmark();
            //}

            ImGui::End();
        }
    }

   private:
    std::shared_ptr<ColoredAsset> TheAsset;
    std::vector<vec2> extras;
    int lightCount  = 128;
    float lightSize = 1;

    SimpleAssetObject assetObject;
    Saiga::AABB bb;

    std::vector<std::shared_ptr<PointLight>> pointLights;

    vec4 start_position;
    quat start_rot;

    vec4 middle_position;
    quat middle_rot;

    vec4 goal_position;
    quat goal_rot;

#ifdef MEDIAN
    int measureFrameCount = 60;
#else
    int measureFrameCount = 300;
#endif
    int currentFrameUpdate = 0;
};

int main(const int argc, const char* argv[])
{
    initSaigaSample();
    Sample example;
    example.run();
    return 0;
}
