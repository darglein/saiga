/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// RenderScene's HDR frame (S17, docs/engine_design.md §10.1, §10.5): the scene pass into an fp16 MSAA target, the
// resolve (clay mode: the hardware average; tone mapped: the tone-map weighted custom resolve), post (exposure, tone
// map, output transfer) and ImGui, pre-exposure, sanitizing, sample counts, GPU sections. Headless (offscreen images),
// skipped without a Vulkan 1.3 device; lavapipe in CI.

#include "saiga/core/imgui/imgui.h"
#include "saiga/vulkan/VulkanForwardRenderer.h"
#include "saiga/vulkan/pipeline/Pipeline.h"
#include "saiga/vulkan/scene/MeshImport.h"
#include "saiga/vulkan/scene/RenderScene.h"
#include "saiga/vulkan/window/HeadlessWindow.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <set>

#include "golden_image.h"
#include "vulkan_test_context.h"

using namespace Saiga;
using namespace Saiga::Vulkan;

namespace
{
constexpr int W = 256, H = 128;

// The reference curves of shader/vulkan/scene/color.glsl
double aces(double x)
{
    return std::clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}
double linearToSrgb(double c)
{
    return c <= 0.0031308 ? c * 12.92 : 1.055 * std::pow(c, 1.0 / 2.4) - 0.055;
}
double srgbToLinear(double c)
{
    return c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4);
}
// The final 8-bit value of a linear (pre-exposure removed) radiance with ACES on a UNORM target
int acesPixel(double radiance)
{
    return int(std::lround(255.0 * linearToSrgb(aces(radiance))));
}

// The gradient of the test pattern at pixel column x: exponential from 16 / 4096 to 16 over the image width (the
// whole tone curve, 12 stops)
double gradient(int x)
{
    return 16.0 * std::exp2(-12.0 * (1.0 - x / 255.0));
}

uint32_t floatBits(float f)
{
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

/**
 * The test draws (pre-exposure aware): triangles in pixel coordinates with either the HDR gradient (gradient()) or a
 * constant value (given as float bits: NaN, inf, ...). In the linear mode (RenderScene::sceneWritesLinear) the value is
 * multiplied by the pre-exposure, otherwise it is written as a display value (clay mode).
 */
class TestPattern
{
   public:
    struct Push
    {
        float v[6];
        float preExposure = 1;
        int32_t mode      = 0;  // 0: gradient, 1: constant (bits), 2: gradient clamped to [0, 1]
        uint32_t bits     = 0;
        int32_t linear    = 0;
    };

    void init(VulkanBase& base, const RenderingFormats& formats)
    {
        static const std::string vert = R"(#version 450
layout(push_constant) uniform Push { vec2 v0; vec2 v1; vec2 v2; float preExposure; int mode; uint bits; int linear; } pc;
void main()
{
    vec2 p      = gl_VertexIndex == 0 ? pc.v0 : (gl_VertexIndex == 1 ? pc.v1 : pc.v2);
    gl_Position = vec4(p, 0.5, 1.0);
}
)";
        static const std::string frag = R"(#version 450
layout(push_constant) uniform Push { vec2 v0; vec2 v1; vec2 v2; float preExposure; int mode; uint bits; int linear; } pc;
layout(location = 0) out vec4 outColor;
void main()
{
    vec3 c;
    if (pc.mode == 1)
        c = vec3(uintBitsToFloat(pc.bits));
    else
        c = vec3(16.0 * exp2(-12.0 * (1.0 - (gl_FragCoord.x - 0.5) / 255.0)));
    if (pc.mode == 2) c = clamp(c, 0.0, 1.0);
    if (pc.linear != 0) c *= pc.preExposure;
    outColor = vec4(c, 1.0);
}
)";
        pipeline.init(base, 0);
        pipeline.addPushConstantRange(
            {vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, uint32_t(sizeof(Push))});
        ASSERT_TRUE(pipeline.shaderPipeline.loadGLSLStrings(
            base.device, {{vk::ShaderStageFlagBits::eVertex, vert}, {vk::ShaderStageFlagBits::eFragment, frag}}));
        PipelineInfo info;
        info.rasterizationState.cullMode        = vk::CullModeFlagBits::eNone;
        info.depthStencilState.depthTestEnable  = VK_FALSE;
        info.depthStencilState.depthWriteEnable = VK_FALSE;
        pipeline.create(formats, info);
    }
    void destroy() { pipeline.destroy(); }

    // A triangle (pixel coordinates)
    void triangle(vk::CommandBuffer cmd, vec2 a, vec2 b, vec2 c, Push push)
    {
        vec2 p[3] = {a, b, c};
        for (int i = 0; i < 3; ++i)
        {
            push.v[2 * i]     = 2.0f * p[i](0) / W - 1.0f;
            push.v[2 * i + 1] = 2.0f * p[i](1) / H - 1.0f;
        }
        if (!pipeline.bind(cmd)) return;
        pipeline.pushConstant(cmd, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, sizeof(Push),
                              &push);
        cmd.draw(3, 1, 0, 0);
    }
    // A rectangle [x0, x1] x [y0, y1] (pixel coordinates)
    void rect(vk::CommandBuffer cmd, float x0, float y0, float x1, float y1, const Push& push)
    {
        triangle(cmd, vec2(x0, y0), vec2(x1, y0), vec2(x1, y1), push);
        triangle(cmd, vec2(x0, y0), vec2(x1, y1), vec2(x0, y1), push);
    }

    Push push(int mode, float value, bool linear, float preExposure) const
    {
        Push p;
        p.mode        = mode;
        p.bits        = floatBits(value);
        p.linear      = linear ? 1 : 0;
        p.preExposure = preExposure;
        return p;
    }

    Pipeline pipeline;
};

// A headless window that can be "minimized" (a 0 x 0 framebuffer: the renderer submits no frame)
class MinimizableWindow : public HeadlessWindow
{
   public:
    using HeadlessWindow::HeadlessWindow;
    ivec2 getFramebufferSize() override { return minimized ? ivec2(0, 0) : HeadlessWindow::getFramebufferSize(); }
    bool minimized = false;
};

// The edge triangle of the bottom half (slanted edges with partially covered pixels)
const vec2 edgeA(20, 124), edgeB(236, 124), edgeC(128, 70);
// A background pixel (the clear value), a gradient pixel (value gradient(midX) = 0.253)
constexpr int bgX = 5, bgY = 100, midX = 128, midY = 30;
}  // namespace

class VulkanScene : public ::testing::Test
{
   protected:
    void SetUp() override
    {
        if (!Instance::suitableDeviceAvailable()) GTEST_SKIP() << "No Vulkan 1.3 device available";
    }
    void TearDown() override { Destroy(); }

    void Create(RenderSceneParameters params, vk::Format offscreen = vk::Format::eR8G8B8A8Unorm)
    {
        Destroy();
        WindowParameters wp;
        wp.width  = W;
        wp.height = H;
        wp.saigaParameters.shaderDirectory.push_back(SAIGA_PROJECT_SOURCE_DIR "/shader");
        wp.imguiParameters.enable = params.imgui;
        VulkanParameters vp;
        vp.enableImgui     = params.imgui;
        vp.offscreenFormat = offscreen;
        window             = std::make_unique<MinimizableWindow>(wp);
        scene              = std::make_unique<RenderScene>(*window, vp, params);
        pattern.init(scene->base(), scene->formats());
    }
    void Destroy()
    {
        if (scene) scene->waitIdle();
        pattern.destroy();
        scene.reset();
        window.reset();
    }

    std::string deviceName() const
    {
        return std::string(scene->base().physicalDevice.getProperties().deviceName.data());
    }

    // Renders one frame with the draw and returns it; preExposures gets RenderScene::preExposure() seen in transfer
    TemplatedImage<ucvec4> Frame(const std::function<void(vk::CommandBuffer, bool linear, float pre)>& draw,
                                 const std::function<void()>& gui = nullptr)
    {
        TemplatedImage<ucvec4> img;
        bool linear   = false;
        float pre     = 0;
        auto transfer = [&](vk::CommandBuffer)
        {
            linear = scene->sceneWritesLinear();
            pre    = scene->preExposure();
            preExposures.push_back(pre);
        };
        auto render = [&](vk::CommandBuffer cmd) { draw(cmd, linear, pre); };
        scene->requestScreenshot([&](const TemplatedImage<ucvec4>& shot) { img = shot; });
        EXPECT_TRUE(scene->renderFrame(transfer, render, gui));
        return img;
    }

    // The gradient in the top half and a bright (value) triangle with slanted edges in the bottom half
    void DrawPattern(vk::CommandBuffer cmd, bool linear, float pre, float bright = 16)
    {
        pattern.rect(cmd, 0, 0, W, 64, pattern.push(0, 0, linear, pre));
        pattern.triangle(cmd, edgeA, edgeB, edgeC, pattern.push(1, bright, linear, pre));
    }

    std::unique_ptr<MinimizableWindow> window;
    std::unique_ptr<RenderScene> scene;
    TestPattern pattern;
    std::vector<float> preExposures;
};

// The tone-mapped frame: the HDR gradient (16 / 4096 .. 16) and a bright-on-dark MSAA edge with ACES, pinned by a
// golden (§17.6)
TEST_F(VulkanScene, AcesGolden)
{
    RenderSceneParameters p;
    p.imgui       = false;
    p.msaaSamples = 4;
    p.toneMap     = ToneMap::Aces;
    p.clearColor  = vec4(0.05f, 0.05f, 0.05f, 1);
    Create(p);
    auto img = Frame([&](vk::CommandBuffer cmd, bool linear, float pre) { DrawPattern(cmd, linear, pre); });
    ASSERT_EQ(img.w, W);
    ASSERT_EQ(img.h, H);
    // The gradient: monotonic, ACES with the sRGB curve (fp16: within 1/255)
    for (int x = 0; x < W; x += 15)
    {
        EXPECT_NEAR(int(img(midY, x)(0)), acesPixel(gradient(x)), 1) << x;
        if (x > 0)
        {
            EXPECT_GE(int(img(midY, x)(0)), int(img(midY, x - 15)(0)));
        }
    }
    // The background is the clear color: a display value, linear for the tone map
    EXPECT_NEAR(int(img(bgY, bgX)(0)), acesPixel(srgbToLinear(0.05)), 1);
    compareGolden(img, "hdr_tonemap", deviceName());
}

// Identity (clay): the gradient clamps, the edge is the plain (hardware) average. ACES: the tone-map weighted resolve
// keeps an edge pixel darker than the plain average would; a tone map switch between frames is consistent
TEST_F(VulkanScene, ResolveAndToneMapSwitch)
{
    RenderSceneParameters p;
    p.imgui       = false;
    p.msaaSamples = 4;
    p.clearColor  = vec4(0, 0, 0, 1);
    Create(p);
    const int n = int(scene->formats().samples);
    if (n == 1) GTEST_SKIP() << "no MSAA";

    // Identity with a bright value of 1: the covered samples per pixel (k = value * n)
    auto identity = Frame([&](vk::CommandBuffer cmd, bool linear, float pre) { DrawPattern(cmd, linear, pre, 1); });
    ASSERT_EQ(identity.w, W);
    EXPECT_EQ(int(identity(midY, 200)(0)), 255);  // gradient(200) = 2.66: clamped
    EXPECT_NEAR(int(identity(midY, midX)(0)), int(std::lround(255 * gradient(midX))), 1);

    scene->setToneMap(ToneMap::Aces);
    EXPECT_FALSE(scene->sceneWritesLinear());  // from the next frame on
    auto aces16 = Frame([&](vk::CommandBuffer cmd, bool linear, float pre) { DrawPattern(cmd, linear, pre, 16); });
    ASSERT_EQ(aces16.w, W);
    EXPECT_TRUE(scene->sceneWritesLinear());
    // The frame after the switch: linear and tone mapped once (no double or missing encoding)
    EXPECT_NEAR(int(aces16(midY, midX)(0)), acesPixel(gradient(midX)), 1);

    int edges = 0;
    for (int y = 64; y < H; ++y)
    {
        for (int x = 0; x < W; ++x)
        {
            const int v = identity(y, x)(0);
            const int k = int(std::lround(v * n / 255.0));
            ASSERT_NEAR(v, 255.0 * k / n, 1.0) << "the plain average of " << n << " samples at " << x << " " << y;
            if (k == 0 || k == n) continue;
            edges++;
            // Weighted: bright samples weigh 1 / (1 + 16), black ones 1
            const double weighted = (k * 16.0 / 17.0) / (k / 17.0 + (n - k));
            const double plain    = 16.0 * k / n;
            EXPECT_NEAR(int(aces16(y, x)(0)), acesPixel(weighted), 2) << x << " " << y << " k " << k;
            EXPECT_LT(int(aces16(y, x)(0)), acesPixel(plain)) << x << " " << y << " k " << k;
        }
    }
    EXPECT_GT(edges, 20);

    // Back to Identity: display values again in the next frame
    scene->setToneMap(ToneMap::Identity);
    auto back = Frame([&](vk::CommandBuffer cmd, bool linear, float pre) { DrawPattern(cmd, linear, pre, 1); });
    ASSERT_EQ(back.w, W);
    for (int y = 0; y < H; y += 7)
        for (int x = 0; x < W; x += 5) EXPECT_EQ(back(y, x), identity(y, x)) << x << " " << y;
}

// Pre-exposure: the exposure changes from the next frame on, the scene of that frame was written with the previous
// exposure, post scales by exposure / preExposure: the result is the same in every frame after the change (no flash)
TEST_F(VulkanScene, ExposureWithoutFlash)
{
    RenderSceneParameters p;
    p.imgui       = false;
    p.msaaSamples = 4;
    p.toneMap     = ToneMap::Aces;
    p.clearColor  = vec4(0.2f, 0.2f, 0.2f, 1);
    Create(p);
    auto draw  = [&](vk::CommandBuffer cmd, bool linear, float pre) { DrawPattern(cmd, linear, pre); };
    auto first = Frame(draw);
    ASSERT_EQ(first.w, W);
    const double bg = srgbToLinear(0.2);
    EXPECT_NEAR(int(first(midY, midX)(0)), acesPixel(gradient(midX)), 1);
    EXPECT_NEAR(int(first(bgY, bgX)(0)), acesPixel(bg), 1);

    scene->setExposure(2);
    preExposures.clear();
    for (int i = 0; i < 3; ++i)
    {
        auto img = Frame(draw);
        ASSERT_EQ(img.w, W);
        EXPECT_NEAR(int(img(midY, midX)(0)), acesPixel(2 * gradient(midX)), 1) << i;
        EXPECT_NEAR(int(img(bgY, bgX)(0)), acesPixel(2 * bg), 1) << i;
        EXPECT_FLOAT_EQ(scene->exposure(), 2.0f);
    }
    ASSERT_EQ(preExposures.size(), 3u);
    EXPECT_FLOAT_EQ(preExposures[0], 1.0f);
    EXPECT_FLOAT_EQ(preExposures[1], 2.0f);
    EXPECT_FLOAT_EQ(preExposures[2], 2.0f);

    // Identity ignores the exposure: no pixel changes
    scene->setToneMap(ToneMap::Identity);
    scene->setExposure(1);
    auto a = Frame(draw);
    a      = Frame(draw);
    scene->setExposure(2);
    auto b = Frame(draw);
    auto c = Frame(draw);
    EXPECT_FLOAT_EQ(scene->exposure(), 1.0f);
    EXPECT_FLOAT_EQ(scene->preExposure(), 1.0f);
    ASSERT_EQ(a.w, W);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
        {
            ASSERT_EQ(a(y, x), b(y, x)) << x << " " << y;
            ASSERT_EQ(a(y, x), c(y, x)) << x << " " << y;
        }
}

// Regression (CodeRabbit on S17): an exposure jump of 17 stops (1024 -> 1/128) pre-exposed with the previous exposure
// wrote radiance 128 as 131072, beyond fp16's 65504: the first frame at the new exposure was darker than the next
// (ACES of 0.5 instead of 1). A change by more than kMaxPreExposureChange pre-exposes with the new exposure.
TEST_F(VulkanScene, LargeExposureChangeWithoutFlash)
{
    RenderSceneParameters p;
    p.imgui       = false;
    p.msaaSamples = 1;
    p.toneMap     = ToneMap::Aces;
    p.exposure    = 1024;
    Create(p);
    auto draw = [&](vk::CommandBuffer cmd, bool linear, float pre)
    { pattern.rect(cmd, 0, 0, W, H, pattern.push(1, 128, linear, pre)); };
    auto bright = Frame(draw);
    ASSERT_EQ(bright.w, W);
    EXPECT_EQ(int(bright(bgY, bgX)(0)), 255);

    scene->setExposure(1.0f / 128);
    preExposures.clear();
    auto first  = Frame(draw);
    auto second = Frame(draw);
    ASSERT_EQ(first.w, W);
    ASSERT_EQ(second.w, W);
    ASSERT_EQ(preExposures.size(), 2u);
    EXPECT_FLOAT_EQ(preExposures[0], 1.0f / 128);
    EXPECT_FLOAT_EQ(preExposures[1], 1.0f / 128);
    EXPECT_NEAR(int(second(bgY, bgX)(0)), acesPixel(1.0), 1);
    EXPECT_EQ(first(bgY, bgX), second(bgY, bgX));
    EXPECT_EQ(first(midY, midX), second(midY, midX));
}

// Regression (CodeRabbit on S17): renderFrame latched the tone map and exposure even when it submitted no frame (a
// minimized window), so the getters described a frame that was never recorded. They are latched when a frame's
// recording starts: the last frame's values stay while minimized, the request applies to the first frame after it.
TEST_F(VulkanScene, LatchesOnlyRecordedFrames)
{
    RenderSceneParameters p;
    p.imgui       = false;
    p.msaaSamples = 1;
    Create(p);
    auto draw = [&](vk::CommandBuffer cmd, bool linear, float pre) { DrawPattern(cmd, linear, pre); };
    ASSERT_EQ(Frame(draw).w, W);

    scene->setToneMap(ToneMap::Aces);
    scene->setExposure(2);
    window->minimized = true;
    bool called       = false;
    auto transfer     = [&](vk::CommandBuffer) { called = true; };
    EXPECT_FALSE(scene->renderFrame(transfer, [](vk::CommandBuffer) {}, nullptr));
    EXPECT_FALSE(called);
    EXPECT_EQ(scene->toneMap(), ToneMap::Identity);
    EXPECT_FALSE(scene->sceneWritesLinear());
    EXPECT_FLOAT_EQ(scene->exposure(), 1.0f);
    EXPECT_FLOAT_EQ(scene->preExposure(), 1.0f);

    window->minimized = false;
    auto img          = Frame(draw);
    ASSERT_EQ(img.w, W);
    EXPECT_EQ(scene->toneMap(), ToneMap::Aces);
    EXPECT_TRUE(scene->sceneWritesLinear());
    EXPECT_FLOAT_EQ(scene->exposure(), 2.0f);
    EXPECT_NEAR(int(img(midY, midX)(0)), acesPixel(2 * gradient(midX)), 1);
}

// NaN, +inf and negative samples give black, white and black pixels (sanitized in the resolve and in post) and do not
// spread to their neighbours, with 1 and 4 samples
TEST_F(VulkanScene, SanitizesSpecialValues)
{
    for (int samples : {1, 4})
    {
        RenderSceneParameters p;
        p.imgui       = false;
        p.msaaSamples = samples;
        p.toneMap     = ToneMap::Aces;
        p.clearColor  = vec4(0.2f, 0.2f, 0.2f, 1);
        Create(p);
        const float values[3] = {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(), -1};
        const int expected[3] = {0, 255, 0};
        auto img              = Frame(
            [&](vk::CommandBuffer cmd, bool linear, float pre)
            {
                for (int i = 0; i < 3; ++i)
                {
                    // Exactly pixel (240 + 4 i, 70): all its samples, none of the neighbours'
                    const float x = 240.0f + 4 * i;
                    pattern.rect(cmd, x, 70, x + 1, 71, pattern.push(1, values[i], linear, pre));
                }
            });
        ASSERT_EQ(img.w, W);
        const int bg = acesPixel(srgbToLinear(0.2));
        for (int i = 0; i < 3; ++i)
        {
            const int x = 240 + 4 * i;
            for (int c = 0; c < 3; ++c) EXPECT_EQ(int(img(70, x)(c)), expected[i]) << samples << " samples, " << i;
            for (auto d : {ivec2(-1, 0), ivec2(1, 0), ivec2(0, -1), ivec2(0, 1)})
                EXPECT_NEAR(int(img(70 + d(1), x + d(0))(0)), bg, 1) << samples << " samples, neighbour of " << i;
        }
    }
}

// A request above the device limits runs at the largest count of the intersection of the framebuffer color and depth
// counts, the sampled image counts and the fp16 format's counts (computed here from the same queries)
TEST_F(VulkanScene, SampleCount)
{
    RenderSceneParameters p;
    p.imgui       = false;
    p.msaaSamples = 64;
    p.toneMap     = ToneMap::Aces;
    Create(p);
    auto pd     = scene->base().physicalDevice;
    auto limits = pd.getProperties().limits;
    auto props =
        pd.getImageFormatProperties(vk::Format::eR16G16B16A16Sfloat, vk::ImageType::e2D, vk::ImageTiling::eOptimal,
                                    vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled);
    uint32_t mask    = uint32_t(limits.framebufferColorSampleCounts & limits.framebufferDepthSampleCounts &
                                limits.sampledImageColorSampleCounts & props.sampleCounts);
    uint32_t largest = 1;
    for (uint32_t c = 1; c <= 64; c *= 2)
        if (mask & c) largest = c;
    EXPECT_EQ(uint32_t(scene->formats().samples), largest);
    EXPECT_EQ(scene->formats().color.size(), 1u);
    EXPECT_EQ(scene->formats().color[0], vk::Format::eR16G16B16A16Sfloat);
    EXPECT_EQ(scene->formats().depth, vk::Format::eD32Sfloat);
    EXPECT_EQ(scene->colorFormat(), vk::Format::eR8G8B8A8Unorm);
    auto img = Frame([&](vk::CommandBuffer cmd, bool linear, float pre) { DrawPattern(cmd, linear, pre); });
    EXPECT_EQ(img.w, W);
    // 1 sample: no resolve
    p.msaaSamples = 1;
    Create(p);
    EXPECT_EQ(scene->formats().samples, vk::SampleCountFlagBits::e1);
    img = Frame([&](vk::CommandBuffer cmd, bool linear, float pre) { DrawPattern(cmd, linear, pre); });
    EXPECT_EQ(img.w, W);
    EXPECT_NEAR(int(img(midY, midX)(0)), acesPixel(gradient(midX)), 1);
}

// The GPU sections are written in every frame and in frame order, also with Identity, 1 sample and hidden ImGui
TEST_F(VulkanScene, GpuSections)
{
    const auto& names = RenderScene::gpuSectionNames();
    ASSERT_EQ(names.size(), 6u);
    EXPECT_STREQ(names[0], "transfer");
    EXPECT_STREQ(names[1], "cull");
    EXPECT_STREQ(names[2], "scene");
    EXPECT_STREQ(names[3], "resolve");
    EXPECT_STREQ(names[4], "post");
    EXPECT_STREQ(names[5], "imgui");
    struct Config
    {
        ToneMap toneMap;
        int samples;
        bool imguiVisible;
    };
    for (Config cfg : {Config{ToneMap::Identity, 1, false}, Config{ToneMap::Identity, 4, true},
                       Config{ToneMap::Aces, 4, true}, Config{ToneMap::Aces, 1, false}})
    {
        RenderSceneParameters p;
        p.imgui       = true;
        p.msaaSamples = cfg.samples;
        p.toneMap     = cfg.toneMap;
        Create(p);
        scene->setImguiVisible(cfg.imguiVisible);
        int got = 0;
        for (int i = 0; i < 8; ++i)
        {
            Frame([&](vk::CommandBuffer cmd, bool linear, float pre) { DrawPattern(cmd, linear, pre); },
                  []() { ImGui::Text("hello"); });
            FrameTimes t;
            while (scene->takeGpuFrameTimes(t))
            {
                got++;
                ASSERT_EQ(t.gpuSectionMs.size(), 6u);
                for (double ms : t.gpuSectionMs)
                {
                    EXPECT_GE(ms, 0);
                    // The frame spans the sections (first begin to last end)
                    EXPECT_GE(t.gpuMs + 1e-6, ms);
                }
            }
        }
        EXPECT_GT(got, 3) << "the GPU times of the frames arrive";
    }
}

// Identity on an sRGB and on a UNORM target gives the same display values (post decodes, the hardware encodes)
TEST_F(VulkanScene, SrgbAndUnormTargetsMatch)
{
    RenderSceneParameters p;
    p.imgui       = false;
    p.msaaSamples = 4;
    auto draw     = [&](vk::CommandBuffer cmd, bool linear, float pre) { DrawPattern(cmd, linear, pre, 0.7f); };
    Create(p, vk::Format::eR8G8B8A8Unorm);
    auto unorm = Frame(draw);
    Create(p, vk::Format::eR8G8B8A8Srgb);
    EXPECT_EQ(scene->colorFormat(), vk::Format::eR8G8B8A8Srgb);
    auto srgb = Frame(draw);
    ASSERT_EQ(unorm.w, W);
    ASSERT_EQ(srgb.w, W);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            for (int c = 0; c < 3; ++c) ASSERT_NEAR(int(unorm(y, x)(c)), int(srgb(y, x)(c)), 1) << x << " " << y;
    // ACES on the sRGB target: linear output, encoded by the hardware (no double encoding)
    p.toneMap = ToneMap::Aces;
    Create(p, vk::Format::eR8G8B8A8Srgb);
    auto aces = Frame(draw);
    ASSERT_EQ(aces.w, W);
    EXPECT_NEAR(int(aces(midY, midX)(0)), acesPixel(gradient(midX)), 1);
}

// Clay mode through RenderScene draws what the forward renderer draws (display values clamped by the shader; the
// demos' MeshRenderer has no golden, this covers its path)
TEST_F(VulkanScene, ClayMatchesForwardRenderer)
{
    const vec4 clear(0.3f, 0.3f, 0.3f, 1);
    auto draw = [&](vk::CommandBuffer cmd, bool linear, float pre)
    {
        pattern.rect(cmd, 0, 0, W, 64, pattern.push(2, 0, linear, pre));
        pattern.triangle(cmd, edgeA, edgeB, edgeC, pattern.push(1, 0.9f, linear, pre));
    };

    RenderSceneParameters p;
    p.imgui       = false;
    p.msaaSamples = 4;
    p.clearColor  = clear;
    Create(p);
    auto viaScene      = Frame(draw);
    const auto samples = scene->formats().samples;
    Destroy();

    WindowParameters wp;
    wp.width  = W;
    wp.height = H;
    wp.saigaParameters.shaderDirectory.push_back(SAIGA_PROJECT_SOURCE_DIR "/shader");
    VulkanParameters vp;
    vp.enableImgui = false;
    ForwardRenderingParameters fp;
    fp.msaaSamples = int(samples);
    fp.reversedZ   = true;
    fp.clearColor  = clear;
    auto fwindow   = std::make_unique<HeadlessWindow>(wp);
    auto forward   = std::make_unique<VulkanForwardRenderer>(*fwindow, vp, fp);
    ASSERT_EQ(forward->sampleCount(), samples);
    VulkanTestApp app;
    forward->setRenderObject(app);
    fwindow->setUpdateObject(app);
    pattern.init(forward->base(), forward->renderingFormats());
    app.onRender = [&](vk::CommandBuffer cmd) { draw(cmd, false, 1); };
    TemplatedImage<ucvec4> viaForward;
    forward->requestScreenshot([&](const TemplatedImage<ucvec4>& img) { viaForward = img; });
    fwindow->renderFrames(1);
    forward->waitIdle();
    pattern.destroy();
    forward.reset();
    fwindow.reset();

    ASSERT_EQ(viaScene.w, W);
    ASSERT_EQ(viaForward.w, W);
    int maxDiff = 0;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            for (int c = 0; c < 3; ++c)
                maxDiff = std::max(maxDiff, std::abs(int(viaScene(y, x)(c)) - int(viaForward(y, x)(c))));
    // Rounding after instead of before the average: at most 1/255 on edge pixels
    EXPECT_LE(maxDiff, 1);
}

//==================================================================================================
// Scene content (S18, docs/engine_design.md §10.1, §10.2, §10.8): meshes, materials, static instance sets, dynamic
// instances, the offset table, the GPU culling (count, scan, write) and the indirect draws, against a CPU reference of
// the culling rule (§17.5: equal except for instances within 1e-3 relative of a plane or an end distance).

namespace
{
constexpr double kPi = 3.14159265358979323846;

// The six faces of a box (CCW seen from outside): the corner signs per face
struct BoxFace
{
    vec3 normal;
    std::array<vec3, 4> corners;
};
const std::array<BoxFace, 6>& boxFaces()
{
    static const std::array<BoxFace, 6> faces = {
        BoxFace{vec3(0, 1, 0), {vec3(-1, 1, 1), vec3(1, 1, 1), vec3(1, 1, -1), vec3(-1, 1, -1)}},
        BoxFace{vec3(0, -1, 0), {vec3(-1, -1, -1), vec3(1, -1, -1), vec3(1, -1, 1), vec3(-1, -1, 1)}},
        BoxFace{vec3(1, 0, 0), {vec3(1, -1, 1), vec3(1, -1, -1), vec3(1, 1, -1), vec3(1, 1, 1)}},
        BoxFace{vec3(-1, 0, 0), {vec3(-1, -1, -1), vec3(-1, -1, 1), vec3(-1, 1, 1), vec3(-1, 1, -1)}},
        BoxFace{vec3(0, 0, 1), {vec3(-1, -1, 1), vec3(1, -1, 1), vec3(1, 1, 1), vec3(-1, 1, 1)}},
        BoxFace{vec3(0, 0, -1), {vec3(1, -1, -1), vec3(-1, -1, -1), vec3(-1, 1, -1), vec3(1, 1, -1)}}};
    return faces;
}

// A box with flat faces (vertices per face) of the half size around center
MeshLod boxLod(const vec3& half, const vec3& center, const ucvec4& color, float endDistance)
{
    MeshLod lod;
    lod.endDistance = endDistance;
    for (const BoxFace& f : boxFaces())
    {
        const uint32_t base = uint32_t(lod.vertices.size());
        for (const vec3& c : f.corners)
        {
            MeshVertex v;
            v.position = center + c.cwiseProduct(half);
            v.normal   = packNormalOct(f.normal);
            v.color    = packColor(color);
            lod.vertices.push_back(v);
        }
        for (uint32_t i : {0u, 1u, 2u, 0u, 2u, 3u}) lod.indices.push_back(base + i);
    }
    return lod;
}

// A mesh of boxes per LOD (one color per LOD), the sphere around them
MeshLods boxMesh(const vec3& half, const vec3& center, const std::vector<float>& endDistances,
                 const std::vector<ucvec4>& colors, float padding = 0)
{
    MeshLods m;
    for (size_t i = 0; i < endDistances.size(); ++i)
        m.lods.push_back(boxLod(half, center, colors[std::min(i, colors.size() - 1)], endDistances[i]));
    m.sphereCenter  = center;
    m.sphereRadius  = half.norm();
    m.boundsPadding = padding;
    return m;
}

// The float quaternion of the rotation by angle about the (unit) axis
quat axisAngle(const vec3& axis, float angle)
{
    const vec3 a = axis.normalized() * std::sin(angle / 2);
    return quat(std::cos(angle / 2), a(0), a(1), a(2));
}

StaticInstance makeInstance(const vec3& position, const quat& rotation, float scale, MeshId mesh)
{
    StaticInstance s;
    s.setPosition(position);
    s.setRotation(rotation);
    s.scale = scale;
    s.mesh  = mesh;
    return s;
}

// The camera rotation looking along forward (y up)
Quat lookRotation(const Vec3& forward)
{
    const Vec3 back  = -forward.normalized();
    const Vec3 right = Vec3(0, 1, 0).cross(back).normalized();
    const Vec3 up    = back.cross(right);
    Mat3 R;
    for (int i = 0; i < 3; ++i)
    {
        R(i, 0) = right(i);
        R(i, 1) = up(i);
        R(i, 2) = back(i);
    }
    return make_quat(R).normalized();
}

// The test camera (double) with its projection into the image (pixel coordinates, row 0 at the top)
struct TestCamera
{
    SpaceId space    = 0;
    Vec3 position    = Vec3::Zero();
    Quat rotation    = Quat::Identity();
    double fovy      = 60.0 * kPi / 180.0;
    double nearPlane = 0.1;
    int w = W, h = H;

    void apply(RenderScene& s) const { s.setCamera(space, position, rotation, fovy); }

    // Inward planes (camera-relative): left, right, bottom, top, near
    std::array<Vec4, 5> planes() const
    {
        const Mat3 R      = rotation.matrix();
        const double tanY = std::tan(fovy / 2), tanX = tanY * double(w) / h;
        std::array<Vec4, 5> p;
        const Vec3 n[4] = {Vec3(1, 0, -tanX), Vec3(-1, 0, -tanX), Vec3(0, 1, -tanY), Vec3(0, -1, -tanY)};
        for (int k = 0; k < 4; ++k)
        {
            const Vec3 m = R * n[k].normalized();
            p[k]         = Vec4(m(0), m(1), m(2), 0);
        }
        const Vec3 f = R * Vec3(0, 0, -1);
        p[4]         = Vec4(f(0), f(1), f(2), -nearPlane);
        return p;
    }

    // The pixel position of a world point
    Vec2 project(const Vec3& world) const
    {
        const Vec3 v      = rotation.matrix().transpose() * (world - position);
        const double f    = 1.0 / std::tan(fovy / 2);
        const double ndcX = f / (double(w) / h) * v(0) / -v(2);
        const double ndcY = -f * v(1) / -v(2);
        return Vec2((ndcX + 1) / 2 * w, (ndcY + 1) / 2 * h);
    }
};

// Rotates v by the quaternion in double
Vec3 rotateD(const quat& q, const Vec3& v)
{
    const Vec3 u(q.x(), q.y(), q.z());
    const Vec3 t = 2.0 * u.cross(v);
    return v + double(q.w()) * t + u.cross(t);
}

// The CPU reference of the culling rule: per LOD the counted instances and the ambiguous ones (within 1e-3 relative of
// a plane or an end distance)
struct Reference
{
    std::array<int, 4> perLod{};
    int ambiguous = 0;
    int total() const { return perLod[0] + perLod[1] + perLod[2] + perLod[3]; }

    // An instance at the camera-relative centre with the radius, of a mesh with the end distances
    void add(const Vec3& center, double radius, const std::vector<float>& ends, const std::array<Vec4, 5>& planes)
    {
        const double d = center.norm();
        bool amb       = false;
        for (float e : ends) amb = amb || std::abs(d - e) <= 1e-3 * e;
        bool inside = d <= ends.back();
        for (const Vec4& p : planes)
        {
            const double s = p(0) * center(0) + p(1) * center(1) + p(2) * center(2) + p(3) + radius;
            amb            = amb || std::abs(s) <= 1e-3 * std::max(d, 1.0);
            inside         = inside && s >= 0;
        }
        if (amb) ambiguous++;
        if (!inside) return;
        int lod = 0;
        while (lod + 1 < int(ends.size()) && d > ends[lod]) lod++;
        perLod[lod]++;
    }
};

void expectCounts(const RenderStatistics& s, const Reference& r, const std::string& what)
{
    int total = 0;
    for (int l = 0; l < 4; ++l)
    {
        EXPECT_LE(std::abs(int(s.instancesPerLod[l]) - r.perLod[l]), r.ambiguous)
            << what << ": LOD " << l << " gpu " << s.instancesPerLod[l] << " reference " << r.perLod[l];
        total += int(s.instancesPerLod[l]);
    }
    EXPECT_LE(std::abs(total - r.total()), r.ambiguous) << what;
    EXPECT_GT(r.total(), 0) << what;
}

// The intensity-weighted centroid of the pixels brighter than the threshold (pixel centres at +0.5), -1 if none
Vec2 centroid(const TemplatedImage<ucvec4>& img, int threshold = 8)
{
    double sum = 0, sx = 0, sy = 0;
    for (int y = 0; y < img.h; ++y)
        for (int x = 0; x < img.w; ++x)
        {
            const int v = img(y, x)(0);
            if (v < threshold) continue;
            sum += v;
            sx += v * (x + 0.5);
            sy += v * (y + 0.5);
        }
    if (sum == 0) return Vec2(-1, -1);
    return Vec2(sx / sum, sy / sum);
}

int maxPixelDiff(const TemplatedImage<ucvec4>& a, const TemplatedImage<ucvec4>& b)
{
    int d = 0;
    for (int y = 0; y < a.h; ++y)
        for (int x = 0; x < a.w; ++x)
            for (int c = 0; c < 3; ++c) d = std::max(d, std::abs(int(a(y, x)(c)) - int(b(y, x)(c))));
    return d;
}

MaterialData flatMaterial(bool twoSided = false)
{
    MaterialData m;
    m.twoSided = twoSided;
    return m;
}
// Emissive white: the pixel is 255 wherever covered (clay mode), independent of the light
MaterialData markerMaterial()
{
    MaterialData m;
    m.baseColor = vec4(0, 0, 0, 1);
    m.emissive  = vec3(1, 1, 1);
    return m;
}
}  // namespace

class VulkanSceneContent : public VulkanScene
{
   protected:
    void CreateScene(RenderSceneParameters p = RenderSceneParameters())
    {
        p.imgui      = false;
        p.clearColor = vec4(0, 0, 0, 1);
        p.nearPlane  = float(camera.nearPlane);
        Create(p);
    }

    // Renders a frame (the scene's draws only), waits and returns the image; stats: that frame's statistics
    TemplatedImage<ucvec4> SceneFrame(RenderStatistics* stats = nullptr)
    {
        auto img = Frame([](vk::CommandBuffer, bool, float) {});
        scene->waitIdle();
        if (stats)
        {
            *stats = scene->statistics();
            EXPECT_EQ(stats->frame, scene->frames()) << "the statistics of the last frame";
        }
        return img;
    }

    TestCamera camera;
};

// The scene's GPU culling against the CPU reference: 10^4 seeded instances of 3 meshes x 3 LODs in several regions
// (levels 0 and 2), after updateInstances, removeInstance (a tombstone inside the frustum) and removeInstanceSet; sets
// of another space are not drawn; the statistics carry the frame they come from
TEST_F(VulkanSceneContent, CullingCountsMatchReference)
{
    CreateScene();
    const std::vector<std::vector<float>> ends = {{40, 120, 300}, {60, 150, 250}, {30, 90, 400}};
    scene->addMaterial(1, "flat", flatMaterial());
    scene->addMaterial(2, "two_sided", flatMaterial(true));
    for (int m = 0; m < 3; ++m)
        scene->addMesh(
            MeshId(m + 1), "mesh" + std::to_string(m),
            boxMesh(vec3(0.5f + m, 1 + m, 0.5f), vec3(0, 1 + m, 0), ends[m],
                    {ucvec4(200, 50, 50, 255), ucvec4(50, 200, 50, 255), ucvec4(50, 50, 200, 255)}, 0.25f * m),
            MaterialId(m == 2 ? 2 : 1));

    camera.position = Vec3(130.5, 20, 70.25);
    camera.rotation = lookRotation(Vec3(1, -0.15, -0.6));
    camera.apply(*scene);

    struct TestSet
    {
        RegionId region;
        std::vector<StaticInstance> instances;
        InstanceSetHandle handle;
    };
    std::vector<TestSet> sets;
    std::mt19937 rng(18);
    std::uniform_real_distribution<float> u(0, 1);
    auto randomInstances = [&](const RegionId& r, int n)
    {
        const Vec3 origin = regionOrigin(r);
        std::vector<StaticInstance> v;
        for (int i = 0; i < n; ++i)
        {
            // Around the camera (within ±420 m), relative to the region
            const Vec3 world = camera.position + Vec3((u(rng) - 0.5) * 840, -20 + u(rng) * 30, (u(rng) - 0.5) * 840);
            const Vec3 local = world - origin;
            const quat q     = axisAngle(vec3(u(rng) - 0.5f, 1, u(rng) - 0.5f), u(rng) * 6.28f);
            v.push_back(makeInstance(local.cast<float>(), q, 0.5f + 1.5f * u(rng), MeshId(1 + rng() % 3)));
        }
        return v;
    };
    const std::vector<RegionId> regions = {RegionId{0, 0, 0, 0}, RegionId{0, 0, 1, 0}, RegionId{0, 0, -1, 1},
                                           RegionId{0, 2, 0, -1}, RegionId{0, 2, -1, 0}};
    for (const RegionId& r : regions)
    {
        TestSet s;
        s.region    = r;
        s.instances = randomInstances(r, 2000);
        s.handle    = scene->addInstanceSet(r, s.instances);
        sets.push_back(s);
    }
    // Another space: never drawn
    const RegionId other{1, 0, 0, 0};
    auto otherInstances = randomInstances(other, 500);
    scene->addInstanceSet(other, otherInstances);

    auto reference = [&]()
    {
        Reference r;
        const auto planes = camera.planes();
        for (const TestSet& s : sets)
        {
            const Vec3 rel = regionOrigin(s.region) - camera.position;
            for (const StaticInstance& i : s.instances)
            {
                if (i.scale == 0) continue;
                const int m        = i.mesh - 1;
                const vec3 center  = vec3(0, 1 + m, 0);
                const double scale = i.scale;
                const Vec3 c =
                    rel + i.getPosition().cast<double>() + rotateD(i.getRotation(), center.cast<double>() * scale);
                const double rad = (vec3(0.5f + m, 1 + m, 0.5f).norm() + 0.25 * m) * scale;
                r.add(c, rad, ends[m], planes);
            }
        }
        return r;
    };

    RenderStatistics st;
    SceneFrame(&st);
    Reference ref = reference();
    expectCounts(st, ref, "initial");
    EXPECT_LT(ref.ambiguous, ref.total() / 10);
    std::cout << "culling: drawn " << ref.total() << " of " << 5 * 2000 << " (ambiguous " << ref.ambiguous << ")"
              << std::endl;

    // updateInstances: move the first 500 instances of a set
    {
        auto moved = randomInstances(sets[1].region, 500);
        scene->updateInstances(sets[1].handle, 0, moved);
        std::copy(moved.begin(), moved.end(), sets[1].instances.begin());
        SceneFrame(&st);
        expectCounts(st, reference(), "after updateInstances");
    }
    // removeInstance: a tombstone inside the frustum is not counted
    {
        const auto planes = camera.planes();
        int removed       = -1;
        for (int i = 0; i < 2000 && removed < 0; ++i)
        {
            Reference one;
            const StaticInstance& inst = sets[0].instances[i];
            const int m                = inst.mesh - 1;
            const Vec3 c = regionOrigin(sets[0].region) - camera.position + inst.getPosition().cast<double>() +
                           rotateD(inst.getRotation(), Vec3(0, 1 + m, 0) * double(inst.scale));
            one.add(c, (vec3(0.5f + m, 1 + m, 0.5f).norm() + 0.25 * m) * inst.scale, ends[m], planes);
            if (one.total() == 1 && one.ambiguous == 0) removed = i;
        }
        ASSERT_GE(removed, 0);
        const Reference before = reference();
        scene->removeInstance(sets[0].handle, uint32_t(removed));
        sets[0].instances[removed].scale = 0;
        SceneFrame(&st);
        const Reference after = reference();
        EXPECT_EQ(after.total(), before.total() - 1);
        expectCounts(st, after, "after removeInstance");
    }
    // removeInstanceSet
    {
        scene->removeInstanceSet(sets[2].handle);
        sets.erase(sets.begin() + 2);
        SceneFrame(&st);
        expectCounts(st, reference(), "after removeInstanceSet");
    }
    // The other space alone: nothing
    scene->setCamera(1, camera.position, camera.rotation, camera.fovy);
    SceneFrame(&st);
    const int otherCount = int(st.instancesPerLod[0] + st.instancesPerLod[1] + st.instancesPerLod[2]);
    EXPECT_GT(otherCount, 0) << "the other space's set is drawn with a camera in that space";
    camera.apply(*scene);
    SceneFrame(&st);
    expectCounts(st, reference(), "back in space 0");
}

// A chunk index that covered 300 instances in one frame and 10 in the next: no stale scratch entries (counts and pixels
// as a fresh scene)
TEST_F(VulkanSceneContent, PartialChunks)
{
    auto setup = [&]()
    {
        CreateScene();
        scene->addMaterial(1, "flat", flatMaterial());
        scene->addMesh(1, "box", boxMesh(vec3(0.4f, 0.4f, 0.4f), vec3(0, 0, 0), {100}, {ucvec4(220, 180, 40, 255)}), 1);
        camera.position = Vec3(0, 0, 20);
        camera.rotation = Quat::Identity();
        camera.apply(*scene);
    };
    auto grid = [](int n)
    {
        std::vector<StaticInstance> v;
        for (int i = 0; i < n; ++i)
            v.push_back(makeInstance(vec3(-8 + (i % 20) * 0.85f, -4 + (i / 20) * 0.6f, 0), quat::Identity(), 0.5f, 1));
        return v;
    };
    setup();
    auto big = scene->addInstanceSet(RegionId(), grid(300));
    RenderStatistics st;
    SceneFrame(&st);
    EXPECT_EQ(st.instancesPerLod[0], 300u);
    scene->removeInstanceSet(big);
    scene->addInstanceSet(RegionId(), grid(10));
    auto after = SceneFrame(&st);
    EXPECT_EQ(st.instancesPerLod[0], 10u);

    setup();
    scene->addInstanceSet(RegionId(), grid(10));
    auto fresh = SceneFrame(&st);
    EXPECT_EQ(st.instancesPerLod[0], 10u);
    ASSERT_EQ(after.w, W);
    EXPECT_EQ(maxPixelDiff(after, fresh), 0);
}

// The statistics of a frame stay correct across a layout change after it (addMesh shifts the buckets)
TEST_F(VulkanSceneContent, StatisticsAcrossLayoutChanges)
{
    CreateScene();
    scene->addMaterial(1, "two_sided", flatMaterial(true));
    scene->addMaterial(2, "opaque", flatMaterial(false));
    scene->addMesh(5, "a", boxMesh(vec3(0.3f, 0.3f, 0.3f), vec3(0, 0, 0), {10, 30}, {ucvec4(255, 0, 0, 255)}), 1);
    camera.position = Vec3(0, 0, 0);
    camera.apply(*scene);
    std::vector<StaticInstance> v;
    for (int i = 0; i < 6; ++i) v.push_back(makeInstance(vec3(0, 0, -5.0f - 4 * i), quat::Identity(), 1, 5));
    scene->addInstanceSet(RegionId(), v);
    auto img             = Frame([](vk::CommandBuffer, bool, float) {});
    const uint64_t frame = scene->frames();
    // A mesh of a class drawn before (opaque) changes every bucket index before the statistics are read
    scene->addMesh(2, "b", boxMesh(vec3(0.3f, 0.3f, 0.3f), vec3(0, 0, 0), {5, 10, 50}, {ucvec4(0, 255, 0, 255)}), 2);
    scene->waitIdle();
    RenderStatistics st = scene->statistics();
    EXPECT_EQ(st.frame, frame);
    EXPECT_EQ(st.instancesPerLod[0], 2u);  // 5, 9
    EXPECT_EQ(st.instancesPerLod[1], 4u);  // 13, 17, 21, 25 (up to 30)
    EXPECT_EQ(st.instancesPerLod[2], 0u);
    EXPECT_EQ(st.nonEmptyBuckets, 2u);
    // The next frame with the new layout and instances of both meshes
    v.clear();
    for (int i = 0; i < 6; ++i) v.push_back(makeInstance(vec3(1, 0, -4.5f - 4 * i), quat::Identity(), 1, 2));
    scene->addInstanceSet(RegionId(), v);
    SceneFrame(&st);
    EXPECT_EQ(st.instancesPerLod[0], 2u + 1u);  // a: 5, 9; b: 4.6
    EXPECT_EQ(st.instancesPerLod[1], 4u + 1u);  // a: 13 .. 25; b: 8.6
    EXPECT_EQ(st.instancesPerLod[2], 4u);       // b: 12.5 .. 24.5
    EXPECT_EQ(st.nonEmptyBuckets, 5u);
    (void)img;
}

// Rejected: a dynamic in another space, a removed dynamic, an unset dynamic and a dynamic of an unregistered mesh
TEST_F(VulkanSceneContent, Rejections)
{
    CreateScene();
    scene->addMaterial(1, "flat", flatMaterial());
    scene->addMesh(1, "box", boxMesh(vec3(0.3f, 0.3f, 0.3f), vec3(0, 0, 0), {100}, {ucvec4(255, 255, 255, 255)}), 1);
    camera.position = Vec3(10, 1, 10);
    camera.apply(*scene);
    DynamicPose pose;
    pose.position = Vec3(10, 1, 5);
    auto visible  = scene->addDynamic(1);
    scene->setDynamic(visible, ChunkId{0, 0, 0}, pose, pose);
    auto otherSpace = scene->addDynamic(1);
    scene->setDynamic(otherSpace, ChunkId{1, 0, 0}, pose, pose);
    auto removed = scene->addDynamic(1);
    scene->setDynamic(removed, ChunkId{0, 0, 0}, pose, pose);
    scene->removeDynamic(removed);
    auto unset = scene->addDynamic(1);
    (void)unset;
    auto unregistered = scene->addDynamic(7);
    scene->setDynamic(unregistered, ChunkId{0, 0, 0}, pose, pose);
    RenderStatistics st;
    SceneFrame(&st);
    EXPECT_EQ(st.instancesPerLod[0], 1u);
    EXPECT_EQ(st.instancesPerLod[1] + st.instancesPerLod[2] + st.instancesPerLod[3], 0u);
    // Nothing is drawn before the first camera: a new scene
    CreateScene();
    scene->addMaterial(1, "flat", flatMaterial());
    scene->addMesh(1, "box", boxMesh(vec3(0.3f, 0.3f, 0.3f), vec3(0, 0, 0), {100}, {ucvec4(255, 255, 255, 255)}), 1);
    scene->addInstanceSet(RegionId(), {makeInstance(vec3(0, 0, 0), quat::Identity(), 1, 1)});
    SceneFrame(&st);
    EXPECT_EQ(st.instancesPerLod[0], 0u);
    EXPECT_EQ(st.drawCommands, 0u);
}

// Growth of the mesh pool and the instance buffer with live sets and meshes (also with uploads pending in the same
// frame): counts and pixels as with large capacities, validation clean, the old buffers released after their frames
TEST_F(VulkanSceneContent, Growth)
{
    auto build = [&](bool small, bool growInFrames)
    {
        RenderSceneParameters p;
        if (small)
        {
            p.vertexCapacity         = 30;
            p.indexCapacity          = 40;
            p.staticInstanceCapacity = 16;
        }
        CreateScene(p);
        scene->addMaterial(1, "flat", flatMaterial());
        camera.position = Vec3(0, 3, 12);
        camera.rotation = lookRotation(Vec3(0, -0.2, -1));
        camera.apply(*scene);
        std::vector<InstanceSetHandle> handles;
        auto addMeshAndSet = [&](int k)
        {
            scene->addMesh(MeshId(k + 1), "m" + std::to_string(k),
                           boxMesh(vec3(0.2f + 0.05f * k, 0.3f, 0.2f), vec3(0, 0.3f, 0), {50},
                                   {ucvec4(40 * k + 30, 255 - 30 * k, 120, 255)}),
                           1);
            std::vector<StaticInstance> v;
            for (int i = 0; i < 12; ++i)
                v.push_back(makeInstance(vec3(-6 + i, 0, -2.0f * k), quat::Identity(), 1, MeshId(k + 1)));
            handles.push_back(scene->addInstanceSet(RegionId(), v, 14));
        };
        RenderStatistics st;
        for (int k = 0; k < 5; ++k)
        {
            addMeshAndSet(k);
            // Uploads pending: an update of an earlier set before the growth of the next one in the same frame
            if (k > 0)
                scene->updateInstances(handles[k - 1], 12,
                                       {makeInstance(vec3(7, 0, -2.0f * (k - 1)), quat::Identity(), 1, MeshId(k))});
            if (growInFrames) SceneFrame(&st);
        }
        return SceneFrame(&st);
    };
    auto reference            = build(false, true);
    RenderStatistics refStats = scene->statistics();
    for (bool inFrames : {true, false})
    {
        auto img            = build(true, inFrames);
        RenderStatistics st = scene->statistics();
        EXPECT_EQ(st.instancesPerLod, refStats.instancesPerLod) << inFrames;
        EXPECT_EQ(maxPixelDiff(img, reference), 0) << inFrames;
    }
    EXPECT_EQ(refStats.instancesPerLod[0], 5u * 12u + 4u);

    // The old buffers are released after their frames: another growth, then the allocations before and after
    // the frames completed
    auto& memory = scene->base().memory;
    std::vector<StaticInstance> many(100, makeInstance(vec3(0, 0, 0), quat::Identity(), 1, 1));
    scene->addInstanceSet(RegionId(), many);
    // Without a screenshot: the frame is in flight when renderFrame returns (its deferred deletions are pending)
    EXPECT_TRUE(scene->renderFrame([](vk::CommandBuffer) {}, [](vk::CommandBuffer) {}, nullptr));
    const size_t before = memory.allocationCount();
    scene->waitIdle();
    scene->base().processDeferredDeletes();
    const size_t after = memory.allocationCount();
    EXPECT_LT(after, before);
}

// Dynamic instances: prev/curr interpolated with alpha (position, rotation on the shorter arc, q vs -q); culling and
// LOD from the same pose; chunk-relative poses in two chunks
TEST_F(VulkanSceneContent, DynamicInterpolation)
{
    RenderSceneParameters p;
    p.msaaSamples = 4;
    CreateScene(p);
    scene->addMaterial(1, "marker", markerMaterial());
    // A 0.2 m marker, LOD 0 up to 12 m, LOD 1 up to 30 m
    scene->addMesh(1, "marker", boxMesh(vec3(0.1f, 0.1f, 0.1f), vec3(0, 0, 0), {12, 30}, {ucvec4(255, 255, 255, 255)}),
                   1);
    // A marker 1 m off the mesh origin along x (rotations)
    scene->addMesh(2, "arm", boxMesh(vec3(0.1f, 0.1f, 0.1f), vec3(1, 0, 0), {100}, {ucvec4(255, 255, 255, 255)}), 1);
    // A 3 m marker (seen from 150 m)
    scene->addMesh(3, "big", boxMesh(vec3(1.5f, 1.5f, 1.5f), vec3(1, 0, 0), {300}, {ucvec4(255, 255, 255, 255)}), 1);
    camera.position = Vec3(256 + 3, 1, 10);
    camera.rotation = Quat::Identity();
    camera.apply(*scene);

    // Position: from 6 m to 16 m in front of the camera, chunk (1, 0)
    const ChunkId chunk{0, 1, 0};
    DynamicPose a, b;
    a.position = Vec3(1.5, 1.2, 4);
    b.position = Vec3(4.5, 0.4, -6);
    auto h     = scene->addDynamic(1);
    scene->setDynamic(h, chunk, a, b);
    for (float alpha : {0.0f, 0.5f, 1.0f})
    {
        scene->setInterpolation(alpha);
        RenderStatistics st;
        auto img            = SceneFrame(&st);
        const Vec3 local    = a.position + (b.position - a.position) * double(alpha);
        const Vec3 world    = chunkOrigin(chunk) + local;
        const Vec2 expected = camera.project(world);
        const Vec2 got      = centroid(img);
        EXPECT_LT((got - expected).norm(), 1.0)
            << "alpha " << alpha << " got " << got.transpose() << " expected " << expected.transpose();
        const double d = (world - camera.position).norm();
        EXPECT_EQ(st.instancesPerLod[0], d <= 12 ? 1u : 0u) << alpha;
        EXPECT_EQ(st.instancesPerLod[1], d > 12 ? 1u : 0u) << alpha;
    }
    scene->removeDynamic(h);

    // Two chunks: the same local pose in chunk (0, 0) and (1, 0) is 256 m apart (the arm's marker 1 m along x)
    {
        camera.position = Vec3(169, 30, 150);
        camera.rotation = lookRotation(Vec3(0, -0.2, -1));
        camera.apply(*scene);
        scene->setInterpolation(1);
        DynamicPose q;
        q.position = Vec3(40, 2, 0);
        auto h0    = scene->addDynamic(3);
        scene->setDynamic(h0, ChunkId{0, 0, 0}, q, q);
        const Vec2 c0 = centroid(SceneFrame());
        EXPECT_LT((c0 - camera.project(Vec3(41, 2, 0))).norm(), 1.0) << c0.transpose();
        scene->setDynamic(h0, ChunkId{0, 1, 0}, q, q);
        const Vec2 c1 = centroid(SceneFrame());
        EXPECT_LT((c1 - camera.project(Vec3(256 + 41, 2, 0))).norm(), 1.0) << c1.transpose();
        scene->removeDynamic(h0);
    }

    // Rotation: 0° -> 270° about y passes through -45° at alpha 0.5 (the shorter arc); q and -q do not move
    {
        camera.position = Vec3(0, 6, 3);
        camera.rotation = lookRotation(Vec3(0, -1, -0.5));
        camera.apply(*scene);
        DynamicPose r0, r1;
        r1.rotation = Quat(AngleAxis(1.5 * kPi, Vec3(0, 1, 0)));
        auto hr     = scene->addDynamic(2);
        scene->setDynamic(hr, ChunkId{0, 0, 0}, r0, r1);
        scene->setInterpolation(0.5f);
        auto img        = SceneFrame();
        const double s2 = std::sqrt(0.5);
        // -45° about y: (1, 0, 0) -> (cos 45°, 0, sin 45°)
        EXPECT_LT((centroid(img) - camera.project(Vec3(s2, 0, s2))).norm(), 1.0)
            << centroid(img).transpose() << " the long arc would be at "
            << camera.project(Vec3(-s2, 0, -s2)).transpose();

        DynamicPose q0, q1;
        q0.rotation = Quat(AngleAxis(0.7, Vec3(0, 1, 0)));
        q1.rotation = Quat(-q0.rotation.w(), -q0.rotation.x(), -q0.rotation.y(), -q0.rotation.z());
        scene->setDynamic(hr, ChunkId{0, 0, 0}, q0, q1);
        const Vec2 expected = camera.project(Vec3(std::cos(0.7), 0, -std::sin(0.7)));
        for (float alpha : {0.0f, 0.5f, 1.0f})
        {
            scene->setInterpolation(alpha);
            EXPECT_LT((centroid(SceneFrame()) - expected).norm(), 0.5) << alpha;
        }
    }
}

// Precision: a camera at 30 km (dyadic coordinates) renders a level-0 chunk as at the origin; a level-6 region whose
// origin is more than 8 km from its instances near the camera: a marker at 0.1 m with a pixel footprint of at most
// 0.25 mm is within 1 px of the double projection and moves with every 0.25 mm camera step (offset hi + lo)
TEST_F(VulkanSceneContent, Precision)
{
    RenderSceneParameters p;
    p.msaaSamples    = 4;
    camera.nearPlane = 0.01;
    CreateScene(p);
    scene->addMaterial(1, "flat", flatMaterial());
    scene->addMaterial(2, "marker", markerMaterial());
    scene->addMesh(1, "box", boxMesh(vec3(0.5f, 0.5f, 0.5f), vec3(0, 0.5f, 0), {300}, {ucvec4(200, 120, 60, 255)}), 1);
    scene->addMesh(2, "marker", boxMesh(vec3(1e-3f, 1e-3f, 1e-3f), vec3(0, 0, 0), {100}, {ucvec4(255, 255, 255, 255)}),
                   2);
    std::vector<StaticInstance> v;
    for (int i = 0; i < 20; ++i)
        v.push_back(makeInstance(vec3(20 + 10 * (i % 5), 0, 20 + 12 * (i / 5)), axisAngle(vec3(0, 1, 0), 0.3f * i),
                                 1 + 0.1f * i, 1));
    camera.rotation = lookRotation(Vec3(1, -0.3, 1));
    camera.position = Vec3(10.5, 6.25, 8.125);
    camera.apply(*scene);
    auto origin   = scene->addInstanceSet(RegionId{0, 0, 0, 0}, v);
    auto atOrigin = SceneFrame();
    scene->removeInstanceSet(origin);
    // 30720 m = 120 chunks (dyadic)
    scene->addInstanceSet(RegionId{0, 0, 120, -120}, v);
    camera.position += Vec3(30720, 0, -30720);
    camera.apply(*scene);
    auto far = SceneFrame();
    EXPECT_EQ(maxPixelDiff(atOrigin, far), 0) << "the image at 30 km equals the image at the origin";

    // Level 6: origin (16384, 0, 0); the marker 9000.125 m from it, the camera 0.1 m in front of it
    CreateScene(p);
    scene->addMaterial(2, "marker", markerMaterial());
    scene->addMesh(2, "marker", boxMesh(vec3(1e-3f, 1e-3f, 1e-3f), vec3(0, 0, 0), {100}, {ucvec4(255, 255, 255, 255)}),
                   2);
    const RegionId region{0, 6, 1, 0};
    const vec3 local(9000.125f, 3.5f, 9000.25f);
    scene->addInstanceSet(region, {makeInstance(local, quat::Identity(), 1, 2)});
    const Vec3 marker = regionOrigin(region) + local.cast<double>();
    // fovy: the footprint 2 * 0.1 * tan(fovy / 2) / H <= 0.25 mm
    camera.fovy     = 2 * std::atan(0.25e-3 * H / (2 * 0.1));
    camera.rotation = Quat::Identity();
    std::vector<double> xs;
    for (int step = -6; step <= 6; ++step)
    {
        camera.position = marker + Vec3(0.0003 + step * 0.25e-3, 0.0001, 0.1);
        camera.apply(*scene);
        auto img       = SceneFrame();
        const Vec2 got = centroid(img);
        const Vec2 exp = camera.project(marker);
        EXPECT_LT((got - exp).norm(), 1.0)
            << "step " << step << " got " << got.transpose() << " expected " << exp.transpose();
        xs.push_back(got(0));
    }
    // The camera moves right: the marker moves left by about one pixel per step
    for (size_t i = 1; i < xs.size(); ++i) EXPECT_LT(xs[i], xs[i - 1] - 0.5) << i;
}

#if defined(SAIGA_USE_ASSIMP)
namespace
{
std::string sceneData(const std::string& file)
{
    return std::string(SAIGA_PROJECT_SOURCE_DIR "/data/tests/scene/") + file;
}
// The importer's color: linear * linear, the sRGB curve, rounded
ucvec4 importedColor(double r, double g, double b)
{
    auto c = [](double v) { return (unsigned char)std::lround(255 * linearToSrgb(std::clamp(v, 0.0, 1.0))); };
    return ucvec4(c(r), c(g), c(b), 255);
}
}  // namespace

// The flat-color import of a glTF with two LODs (Box_LOD0 with two primitives of two materials, Box_LOD1), a VEC3
// COLOR_0: merged primitives, duplicated faces, face normals, opaque alpha, baked base color factors; palette indices;
// errors (a palette miss, LINES, a LOD gap)
TEST(VulkanMeshImport, FlatBox)
{
    MeshLods lods;
    MaterialData material;
    std::string error;
    ASSERT_TRUE(importMeshLods(sceneData("flat_box.gltf"), lods, &material, error)) << error;
    ASSERT_EQ(lods.lods.size(), 2u);
    EXPECT_EQ(lods.lods[0].vertices.size(), 36u);  // 12 triangles
    EXPECT_EQ(lods.lods[1].vertices.size(), 12u);  // 4 triangles
    EXPECT_LT(lods.lods[0].endDistance, lods.lods[1].endDistance);
    const ucvec4 top   = importedColor(0.8, 0.2, 0.1);
    const ucvec4 side  = importedColor(0.05, 0.3, 0.15);  // COLOR_0 (0.1, 0.6, 0.3) * baseColorFactor 0.5
    const ucvec4 tetra = importedColor(0.2, 0.2, 0.9);
    int topFaces = 0, sideFaces = 0;
    for (size_t l = 0; l < 2; ++l)
    {
        const MeshLod& lod = lods.lods[l];
        ASSERT_EQ(lod.indices.size(), lod.vertices.size());
        for (size_t t = 0; t < lod.vertices.size(); t += 3)
        {
            const vec3 a = lod.vertices[t].position, b = lod.vertices[t + 1].position, c = lod.vertices[t + 2].position;
            const vec3 n = (b - a).cross(c - a).normalized();
            for (int k = 0; k < 3; ++k)
            {
                EXPECT_EQ(lod.indices[t + k], uint32_t(t + k));
                EXPECT_EQ(lod.vertices[t + k].normal, lod.vertices[t].normal);
                EXPECT_GT(unpackNormalOct(lod.vertices[t + k].normal).dot(n), 0.999f);
                EXPECT_EQ(lod.vertices[t + k].color >> 24, 255u) << "opaque";
            }
            // Outward: the face normal points away from the centre (0, 0.5, 0)
            EXPECT_GT(n.dot((a + b + c) / 3 - vec3(0, 0.5f, 0)), 0) << l << " " << t;
            const uint32_t col = lod.vertices[t].color;
            if (l == 1)
            {
                EXPECT_EQ(col, packColor(tetra));
            }
            else if (std::abs(n(1)) > 0.9f)
            {
                EXPECT_EQ(col, packColor(top));
                topFaces++;
            }
            else
            {
                EXPECT_EQ(col, packColor(side));
                sideFaces++;
            }
        }
    }
    EXPECT_EQ(topFaces, 4);
    EXPECT_EQ(sideFaces, 8);
    // The node translation (0, 0.5, 0) is baked
    EXPECT_NEAR(lods.sphereCenter(1), 0.5f, 1e-6f);
    EXPECT_NEAR(lods.sphereRadius, std::sqrt(0.75f), 1e-5f);
    EXPECT_FALSE(material.usePalette);
    EXPECT_FALSE(material.alphaMask);
    EXPECT_FALSE(material.twoSided);
    EXPECT_FLOAT_EQ(material.roughness, 0.7f);
    EXPECT_FLOAT_EQ(material.metalness, 0.0f);
    EXPECT_EQ(material.baseColor, vec4(1, 1, 1, 1));

    // A palette: the indices of the exact RGB entries
    std::vector<ucvec4> palette = {ucvec4(1, 2, 3, 255), side, tetra, top};
    MeshImportOptions options;
    options.palette = &palette;
    ASSERT_TRUE(importMeshLods(sceneData("flat_box.gltf"), lods, &material, error, options)) << error;
    EXPECT_TRUE(material.usePalette);
    for (const MeshVertex& v : lods.lods[1].vertices) EXPECT_EQ(v.color, 2u);
    std::set<uint32_t> used;
    for (const MeshVertex& v : lods.lods[0].vertices) used.insert(v.color);
    EXPECT_EQ(used, (std::set<uint32_t>{1u, 3u}));

    // Errors: a palette miss names the color, LINES, a LOD gap, a missing file
    palette.pop_back();
    EXPECT_FALSE(importMeshLods(sceneData("flat_box.gltf"), lods, &material, error, options));
    EXPECT_NE(error.find("not in the palette"), std::string::npos) << error;
    EXPECT_NE(error.find(std::to_string(int(top(0)))), std::string::npos) << error;
    EXPECT_FALSE(importMeshLods(sceneData("flat_box_lines.gltf"), lods, &material, error));
    EXPECT_NE(error.find("only triangles"), std::string::npos) << error;
    EXPECT_FALSE(importMeshLods(sceneData("flat_box_lod_gap.gltf"), lods, &material, error));
    EXPECT_NE(error.find("not contiguous"), std::string::npos) << error;
    EXPECT_FALSE(importMeshLods(sceneData("missing.gltf"), lods, &material, error));
    EXPECT_FALSE(error.empty());
}

// The imported box rendered as one instance (face-duplicated flat normals, the fixed light), pinned by a golden
TEST_F(VulkanSceneContent, ImportedMeshGolden)
{
    MeshLods lods;
    MaterialData material;
    std::string error;
    ASSERT_TRUE(importMeshLods(sceneData("flat_box.gltf"), lods, &material, error)) << error;
    lods.lods[0].endDistance = 50;
    lods.lods[1].endDistance = 100;
    RenderSceneParameters p;
    p.msaaSamples = 4;
    CreateScene(p);
    scene->addMaterial(1, "box", material);
    scene->addMesh(1, "box", lods, 1);
    camera.position = Vec3(1.6, 2.1, 2.4);
    camera.rotation = lookRotation(Vec3(-1.6, -1.6, -2.4));
    camera.apply(*scene);
    scene->addInstanceSet(RegionId(), {makeInstance(vec3(0, 0, 0), quat::Identity(), 1, 1),
                                       makeInstance(vec3(-3, 0, -3), axisAngle(vec3(0, 1, 0), 0.5f), 1.5f, 1)});
    RenderStatistics st;
    auto img = SceneFrame(&st);
    ASSERT_EQ(img.w, W);
    EXPECT_EQ(st.instancesPerLod[0], 2u);
    // The three visible faces of the near box have three different flat shades (one per face normal)
    const Vec2 c = camera.project(Vec3(0, 0.5, 0));
    const int cx = int(c(0)), cy = int(c(1));
    const auto top   = img(cy - 12, cx);
    const auto left  = img(cy + 8, cx - 10);
    const auto right = img(cy + 8, cx + 10);
    EXPECT_NE(top, left);
    EXPECT_NE(left, right);
    compareGolden(img, "scene_flat_box", deviceName());
}
#endif

// The draw paths with and without multiDrawIndirect give equal counts and images; drawCommands (one per pipeline class
// or one per bucket) and nonEmptyBuckets
TEST_F(VulkanSceneContent, DrawPaths)
{
    TemplatedImage<ucvec4> images[2];
    RenderStatistics stats[2];
    bool multi[2] = {false, false};
    for (int path = 0; path < 2; ++path)
    {
        RenderSceneParameters p;
        p.allowMultiDrawIndirect = path == 1;
        CreateScene(p);
        multi[path] = scene->multiDrawIndirect();
        scene->addMaterial(1, "opaque", flatMaterial());
        scene->addMaterial(2, "two_sided", flatMaterial(true));
        // Two classes: meshes 1, 2 opaque (3 + 2 LODs), mesh 3 two-sided (2 LODs): 7 buckets
        scene->addMesh(1, "a",
                       boxMesh(vec3(0.4f, 0.4f, 0.4f), vec3(0, 0.4f, 0), {8, 16, 40}, {ucvec4(250, 60, 60, 255)}), 1);
        scene->addMesh(2, "b", boxMesh(vec3(0.3f, 0.6f, 0.3f), vec3(0, 0.6f, 0), {12, 40}, {ucvec4(60, 250, 60, 255)}),
                       1);
        scene->addMesh(3, "c", boxMesh(vec3(0.5f, 0.2f, 0.5f), vec3(0, 0.2f, 0), {10, 40}, {ucvec4(60, 60, 250, 255)}),
                       2);
        camera.position = Vec3(0, 2, 6);
        camera.rotation = lookRotation(Vec3(0, -0.15, -1));
        camera.apply(*scene);
        std::vector<StaticInstance> v;
        for (int i = 0; i < 12; ++i)
            v.push_back(
                makeInstance(vec3(-3 + (i % 4) * 2.0f, 0, 2 - 4.0f * (i / 4)), quat::Identity(), 1, MeshId(1 + i % 2)));
        // Mesh 3 only far (LOD 1)
        v.push_back(makeInstance(vec3(0, 0, -12), quat::Identity(), 1, 3));
        scene->addInstanceSet(RegionId(), v);
        images[path] = SceneFrame(&stats[path]);
    }
    EXPECT_FALSE(multi[0]);
    EXPECT_EQ(stats[0].instancesPerLod, stats[1].instancesPerLod);
    EXPECT_EQ(stats[0].nonEmptyBuckets, stats[1].nonEmptyBuckets);
    EXPECT_EQ(maxPixelDiff(images[0], images[1]), 0);
    // One by one: every bucket, empty ones too
    EXPECT_EQ(stats[0].drawCommands, 7u);
    EXPECT_EQ(stats[1].drawCommands, multi[1] ? 2u : 7u);
    // Distances 4.5 .. 12.5: a in LOD 0, 1; b in LOD 0, 1; c in LOD 1
    EXPECT_GE(stats[0].nonEmptyBuckets, 4u);
    EXPECT_EQ(stats[0].instancesPerLod[0] + stats[0].instancesPerLod[1] + stats[0].instancesPerLod[2], 13u);
}
