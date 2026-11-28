/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
// The forward, deferred and uber deferred renderers on a headless window: each renders a small scene with shadowed
// lights, and its passes must not depend on the state an earlier pass or the application left behind.
//
// With SAIGA_RENDERER_TEST_OUTPUT=<dir>, the images are written there as <variant>.png, to compare the renderers'
// output before and after a change. Every test is skipped without an EGL display. On Mesa's software renderer, as CI
// does:
//   __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json ctest -R test_opengl_renderers

#include "saiga/core/framework/framework.h"
#include "saiga/core/math/random.h"
#include "saiga/core/model/model_from_shape.h"
#include "saiga/core/util/env.h"
#include "saiga/opengl/assets/AssetRenderSystem.h"
#include "saiga/opengl/assets/all.h"
#include "saiga/opengl/egl/offscreen_window.h"
#include "saiga/opengl/opengl.h"
#include "saiga/opengl/pipeline.h"
#include "saiga/opengl/rendering/deferredRendering/deferredRendering.h"
#include "saiga/opengl/rendering/deferredRendering/uber_deferred_renderer.h"
#include "saiga/opengl/rendering/forwardRendering/forwardRendering.h"
#include "saiga/opengl/window/StandaloneWindow.h"
#include "saiga/opengl/world/LineSoup.h"
#include "saiga/opengl/world/proceduralSkybox.h"

#include "gtest/gtest.h"

#include <filesystem>
#include <functional>
#include <memory>

using namespace Saiga;

namespace
{
constexpr int width  = 256;
constexpr int height = 192;

enum class Kind
{
    Forward,
    Deferred,
    UberDeferred
};

struct Variant
{
    std::string name;
    Kind kind = Kind::Deferred;
    // The deferred renderer's parameters, set before it is created, and its lighting options.
    std::function<void(DeferredRenderingParameters&)> deferredParams;
    std::function<void(DeferredLighting&)> deferredLighting;
    bool pointAndSpotLights = false;
    bool volumetric         = false;
};

WindowParameters HeadlessParameters(bool imgui = false)
{
    WindowParameters params;
    params.backend = WindowParameters::Backend::Headless;
    params.width   = width;
    params.height  = height;
    // Without ImGui the renderers draw into the whole window instead of the editor layout's 3D view.
    params.imguiParameters.enable = imgui;
    return params;
}

// State that a careless application or library could leave behind, set with gl calls directly.
void SetUnrelatedState()
{
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_FRONT);
    glEnable(GL_SCISSOR_TEST);
    glScissor(0, 0, 1, 1);
    glEnable(GL_STENCIL_TEST);
    glStencilFunc(GL_NEVER, 3, 0x0F);
    glDepthMask(GL_FALSE);
    glDepthFunc(GL_GREATER);
    glColorMask(GL_FALSE, GL_TRUE, GL_FALSE, GL_TRUE);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(4.f, 4.f);
}

class Scene : public StandaloneWindow
{
   public:
    explicit Scene(const Variant& variant, bool imgui = false)
        : StandaloneWindow(HeadlessParameters(imgui), OpenGLParameters())
    {
        // The editor layout would size the 3D view by its ImGui window.
        editor_gui.enabled = false;
        Random::setSeed(3);
        kind = variant.kind;
        switch (variant.kind)
        {
            case Kind::Forward:
            {
                auto r   = CreateRenderer<ForwardRenderer>(ForwardRenderingParameters());
                lighting = &r->lighting;
                renderer = r;
                break;
            }
            case Kind::Deferred:
            {
                DeferredRenderingParameters params;
                if (variant.deferredParams) variant.deferredParams(params);
                auto r = CreateRenderer<DeferredRenderer>(params);
                if (variant.deferredLighting) variant.deferredLighting(r->lighting);
                lighting = &r->lighting;
                renderer = r;
                break;
            }
            case Kind::UberDeferred:
            {
                auto r   = CreateRenderer<UberDeferredRenderer>(UberDeferredRenderingParameters());
                lighting = &r->lighting;
                renderer = r;
                break;
            }
        }

        camera.setProj(60.0f, float(width) / height, 0.1f, 60.0f);
        camera.setView(vec3(0, 6, 12), vec3(0, 0, 0), vec3(0, 1, 0));
        window->setCamera(&camera);

        groundPlane = std::make_shared<ColoredAsset>(
            CheckerBoardPlane(make_ivec2(20, 20), 1.0f, Colors::indianred, Colors::lightgray));
        box = std::make_shared<ColoredAsset>(
            BoxMesh(AABB(vec3(-1, 0, -1), vec3(1, 2, 1))).FlatShading().SetVertexColor(vec4(0.7, 0.7, 0.7, 1)));
        sphere = std::make_shared<ColoredAsset>(
            IcoSphereMesh(Sphere(make_vec3(0), 1), 2).SetVertexColor(vec4(0.3, 0.6, 0.9, 1)));
        for (int i = 0; i < 6; ++i)
        {
            mat4 m = translate(vec3(-6 + 2.5f * i, 0, -2 + (i % 2) * 3));
            boxes.push_back(m);
        }

        lines.lines.push_back({vec3(-5, 0.1f, 4), vec3(1, 1, 0)});
        lines.lines.push_back({vec3(5, 3, 4), vec3(1, 1, 0)});
        lines.updateBuffer();

        sun = std::make_shared<DirectionalLight>();
        lighting->AddLight(sun);
        sun->BuildCascades(3);
        sun->castShadows = true;
        sun->setDirection(vec3(-1, -2, -1.5f));
        sun->setIntensity(variant.pointAndSpotLights ? 0.3f : 1.f);
        sun->fitShadowToCamera(&camera);

        if (variant.pointAndSpotLights)
        {
            auto point = std::make_shared<PointLight>();
            point->setIntensity(3);
            point->setRadius(8);
            point->position = vec3(-2, 3, 2);
            point->setColorDiffuse(vec3(1, 0.8f, 0.6f));
            point->castShadows = true;
            point->volumetric  = variant.volumetric;
            lighting->AddLight(point);

            auto spot = std::make_shared<SpotLight>();
            spot->setIntensity(4);
            spot->setRadius(12);
            spot->setAngle(45);
            spot->position  = vec3(3, 6, 1);
            spot->direction = vec3(0, -1, 0);
            spot->setColorDiffuse(vec3(0.6f, 0.8f, 1));
            spot->castShadows = true;
            spot->volumetric  = variant.volumetric;
            lighting->AddLight(spot);
        }
        lighting->renderVolumetric = variant.volumetric;
    }

    void update(float) override {}
    void interpolate(float, float) override {}

    void render(RenderInfo render_info) override
    {
        render_system.Clear();
        // The forward renderer draws the assets in its forward pass, which only takes the unlit ones.
        const int flags = (kind == Kind::Forward ? RENDER_UNLIT : RENDER_DEFAULT) | RENDER_SHADOW;
        render_system.Add(groundPlane.get(), mat4::Identity(), flags);
        for (auto& m : boxes) render_system.Add(box.get(), m, flags);
        render_system.Add(sphere.get(), translate(vec3(1, 2.5f, 1)), flags);
        render_system.Render(render_info);
        if (render_info.render_pass == RenderPass::Forward)
        {
            skybox.sunDir = sun->getDirection();
            skybox.render(render_info.camera);
            lines.render(render_info.camera);
        }
    }

    // Renders one frame and returns the renderer's output, row 0 at the bottom.
    TemplatedImage<ucvec4> Frame()
    {
        window->render();
        window->swap();
        auto color = renderer->target_framebuffer->getTextureColor(0);
        TemplatedImage<ucvec4> img(color->getHeight(), color->getWidth());
        color->download(img.data8());
        return img;
    }

    Kind kind                  = Kind::Deferred;
    OpenGLRenderer* renderer   = nullptr;
    RendererLighting* lighting = nullptr;
    PerspectiveCamera camera;
    AssetRenderSystem render_system;
    std::shared_ptr<ColoredAsset> groundPlane, box, sphere;
    std::vector<mat4> boxes;
    LineSoup lines;
    ProceduralSkybox skybox;
    std::shared_ptr<DirectionalLight> sun;
};

// The number of pixels in which a and b differ by more than one step in any channel.
int DifferentPixels(const TemplatedImage<ucvec4>& a, const TemplatedImage<ucvec4>& b)
{
    int count = 0;
    for (int y = 0; y < a.h; ++y)
    {
        for (int x = 0; x < a.w; ++x)
        {
            ivec4 d = (a(y, x).cast<int>() - b(y, x).cast<int>()).array().abs();
            if (d.maxCoeff() > 1) ++count;
        }
    }
    return count;
}

// The number of distinct colors, to tell a rendered scene from a cleared or broken image.
int DistinctColors(const TemplatedImage<ucvec4>& img)
{
    std::vector<uint32_t> colors;
    for (int y = 0; y < img.h; y += 4)
    {
        for (int x = 0; x < img.w; x += 4)
        {
            const ucvec4 c = img(y, x);
            colors.push_back(c(0) | (c(1) << 8) | (c(2) << 16));
        }
    }
    std::sort(colors.begin(), colors.end());
    return int(std::unique(colors.begin(), colors.end()) - colors.begin());
}

std::vector<Variant> Variants()
{
    using P     = DeferredRenderingParameters;
    auto params = [](std::function<void(P&)> f) { return f; };
    std::vector<Variant> v;
    v.push_back({"forward", Kind::Forward});
    v.push_back({"forward_lights", Kind::Forward, nullptr, nullptr, true});
    v.push_back({"deferred", Kind::Deferred});
    v.push_back({"deferred_lights", Kind::Deferred, nullptr, nullptr, true});
    v.push_back({"deferred_lights_no_stencil_culling", Kind::Deferred, nullptr,
                 [](DeferredLighting& l) { l.stencilCulling = false; }, true});
    v.push_back({"deferred_lights_no_depth_test", Kind::Deferred, nullptr,
                 [](DeferredLighting& l) { l.lightDepthTest = false; }, true});
    v.push_back({"deferred_volumetric", Kind::Deferred, nullptr, nullptr, true, true});
    v.push_back({"deferred_ldr", Kind::Deferred, params([](P& p) { p.hdr = false; })});
    v.push_back({"deferred_bloom", Kind::Deferred, params([](P& p) { p.bloom = true; })});
    v.push_back({"deferred_ssao", Kind::Deferred, params([](P& p) { p.useSSAO = true; })});
    v.push_back({"deferred_smaa", Kind::Deferred, params([](P& p) { p.useSMAA = true; })});
    v.push_back({"deferred_wireframe", Kind::Deferred, params([](P& p) { p.wireframe = true; })});
    v.push_back({"deferred_depth_offset", Kind::Deferred, params([](P& p) { p.offsetGeometry = true; })});
    v.push_back({"deferred_no_used_pixel_mask", Kind::Deferred, params([](P& p) { p.maskUsedPixels = false; })});
    v.push_back(
        {"deferred_write_depth", Kind::Deferred, params([](P& p) { p.writeDepthToDefaultFramebuffer = true; })});
    v.push_back({"uber_deferred", Kind::UberDeferred});
    v.push_back({"uber_deferred_lights", Kind::UberDeferred, nullptr, nullptr, true});
    return v;
}

void PrintTo(const Variant& variant, std::ostream* os)
{
    *os << variant.name;
}

class OpenGLRenderers : public ::testing::TestWithParam<Variant>
{
   public:
    // The renderers load their shaders from the source tree (without the config file initSaigaSample writes).
    static void SetUpTestSuite()
    {
        SaigaParameters params;
        params.shaderDirectory.push_back(SAIGA_PROJECT_SOURCE_DIR "/shader");
        params.dataDirectory.push_back(SAIGA_PROJECT_SOURCE_DIR "/data");
        params.textureDirectory.push_back(SAIGA_PROJECT_SOURCE_DIR "/data/textures");
        initSaiga(params);
    }

   protected:
    void SetUp() override
    {
        if (!CreateEGLContext()) GTEST_SKIP() << "No EGL display available";
        TerminateEGLContext();
    }
};

TEST_P(OpenGLRenderers, IndependentOfEarlierState)
{
    const Variant& variant = GetParam();
    Scene scene(variant);

    // Some passes build their resources in the first frames.
    for (int i = 0; i < 3; ++i) scene.Frame();
    const TemplatedImage<ucvec4> reference = scene.Frame();
    EXPECT_GT(DistinctColors(reference), 20) << "The scene was not rendered";

    const std::string output = GetEnvIfSet("SAIGA_RENDERER_TEST_OUTPUT").value_or("");
    if (!output.empty())
    {
        std::filesystem::create_directories(output);
        TemplatedImage<ucvec4> flipped = reference;
        flipped.getImageView().flipY();
        flipped.save(output + "/" + variant.name + ".png");
    }

    SetUnrelatedState();
    const TemplatedImage<ucvec4> after = scene.Frame();
    EXPECT_EQ(DifferentPixels(reference, after), 0);
    EXPECT_EQ(glGetError(), GL_NO_ERROR);
}

// With the cache, applyPipelineState skips the values that don't change. The image stays the same, and the validation
// asserts if any pass changed the state without going through applyPipelineState.
TEST_P(OpenGLRenderers, SameImageWithTheCache)
{
    const Variant& variant = GetParam();
    Scene scene(variant);
    for (int i = 0; i < 3; ++i) scene.Frame();
    const TemplatedImage<ucvec4> reference = scene.Frame();

    PipelineStateCache::setEnabled(true);
    PipelineStateCache::setValidation(true);
    for (int i = 0; i < 2; ++i) scene.Frame();
    const TemplatedImage<ucvec4> cached = scene.Frame();
    PipelineStateCache::setEnabled(false);
    PipelineStateCache::setValidation(false);

    EXPECT_EQ(DifferentPixels(reference, cached), 0);
}

// ImGui's OpenGL backend sets its state with gl calls and restores the state it found at the end of its frame, so the
// cache stays valid across it.
TEST(OpenGLRendererImGui, CacheStaysValid)
{
    if (!CreateEGLContext()) GTEST_SKIP() << "No EGL display available";
    TerminateEGLContext();
    OpenGLRenderers::SetUpTestSuite();

    Variant variant{"deferred_lights", Kind::Deferred, nullptr, nullptr, true};
    Scene scene(variant, true);
    for (int i = 0; i < 3; ++i) scene.Frame();
    const TemplatedImage<ucvec4> reference = scene.Frame();

    PipelineStateCache::setEnabled(true);
    PipelineStateCache::setValidation(true);
    for (int i = 0; i < 3; ++i) scene.Frame();
    const TemplatedImage<ucvec4> cached = scene.Frame();
    PipelineStateCache::setEnabled(false);
    PipelineStateCache::setValidation(false);

    EXPECT_EQ(DifferentPixels(reference, cached), 0);
}

INSTANTIATE_TEST_SUITE_P(Renderers, OpenGLRenderers, ::testing::ValuesIn(Variants()),
                         [](const ::testing::TestParamInfo<Variant>& info) { return info.param.name; });

}  // namespace
