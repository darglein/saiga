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
#include "saiga/vulkan/scene/GrassPass.h"
#include "saiga/vulkan/scene/MeshImport.h"
#include "saiga/vulkan/scene/MeshPool.h"
#include "saiga/vulkan/scene/PrototypeWorld.h"
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
#include "terrain_test_hooks.h"
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

// S36 (round-1 blocker 1): with an environment the frame exposure is exposureFromEv100(EV100, multiplier) over the
// whole EV100 range of day and night, unclamped, and a radiance scaled to the matching exposure gives the same post
// output (the pre-exposure keeps the fp16 values near 1)
TEST_F(VulkanScene, Ev100RangeReachesPost)
{
    RenderSceneParameters p;
    p.imgui       = false;
    p.msaaSamples = 4;
    p.toneMap     = ToneMap::Aces;
    Create(p);
    std::vector<int> pixels;
    for (float ev100 : {20.0f, 15.0f, -1.0f, -8.0f})
    {
        EnvironmentState env;
        env.ev100 = ev100;
        scene->setEnvironment(env);
        const float exposure = exposureFromEv100(ev100);
        EXPECT_GT(exposure, std::exp2(-24.0f));
        EXPECT_LT(exposure, std::exp2(10.0f));
        const float radiance = 0.5f / exposure;
        auto draw            = [&](vk::CommandBuffer cmd, bool linear, float pre)
        { pattern.rect(cmd, 0, 0, W, H, pattern.push(1, radiance, linear, pre)); };
        auto img = Frame(draw);
        ASSERT_EQ(img.w, W);
        EXPECT_FLOAT_EQ(scene->frameExposure(), exposure) << ev100;
        EXPECT_FLOAT_EQ(scene->preExposure(), exposure) << ev100;
        EXPECT_FLOAT_EQ(scene->exposure(), 1.0f);
        EXPECT_NEAR(int(img(bgY, bgX)(0)), acesPixel(0.5), 1) << ev100;
        pixels.push_back(int(img(bgY, bgX)(0)));
    }
    for (int px : pixels) EXPECT_NEAR(px, pixels.front(), 1);
}

// S36 (review 2, finding 6): a scene on which setEnvironment was never called keeps the unit-less exposure
TEST_F(VulkanScene, NoEnvironmentKeepsUnitlessExposure)
{
    RenderSceneParameters p;
    p.imgui       = false;
    p.msaaSamples = 1;
    p.toneMap     = ToneMap::Aces;
    p.exposure    = 2;
    Create(p);
    auto draw = [&](vk::CommandBuffer cmd, bool linear, float pre) { DrawPattern(cmd, linear, pre); };
    ASSERT_EQ(Frame(draw).w, W);
    EXPECT_FALSE(scene->hasEnvironment());
    EXPECT_FLOAT_EQ(scene->frameExposure(), 2.0f);
    EXPECT_FLOAT_EQ(scene->exposure(), 2.0f);
    // With an environment the multiplier stays a multiplier on top of the EV100
    EnvironmentState env;
    env.ev100 = 10;
    scene->setEnvironment(env);
    ASSERT_EQ(Frame(draw).w, W);
    EXPECT_TRUE(scene->hasEnvironment());
    EXPECT_FLOAT_EQ(scene->exposure(), 2.0f);
    EXPECT_FLOAT_EQ(scene->frameExposure(), exposureFromEv100(10, 2));
    // Clay ignores both
    scene->setToneMap(ToneMap::Identity);
    ASSERT_EQ(Frame(draw).w, W);
    EXPECT_FLOAT_EQ(scene->frameExposure(), 1.0f);
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
    // S19a: the depth pre-pass and the sub-sections of the passes (dotted names), written in every frame
    const auto& names = RenderScene::gpuSectionNames();
    ASSERT_EQ(names.size(), 16u);
    EXPECT_STREQ(names[0], "transfer");
    // S36a1c: the atmosphere and the environment map
    EXPECT_STREQ(names[1], "environment");
    EXPECT_STREQ(names[2], "cull");
    EXPECT_STREQ(names[3], "depth");
    EXPECT_STREQ(names[4], "depth.terrain");
    EXPECT_STREQ(names[5], "scene");
    EXPECT_STREQ(names[6], "scene.terrain");
    EXPECT_STREQ(names[7], "scene.instances");
    EXPECT_STREQ(names[8], "scene.grass");
    EXPECT_STREQ(names[9], "scene.sky");
    // S36a1c: the 2D cloud layer
    EXPECT_STREQ(names[10], "scene.clouds");
    // S36a1b: the debug lines and the lighting's preview conversions
    EXPECT_STREQ(names[11], "scene.debug");
    EXPECT_STREQ(names[12], "resolve");
    EXPECT_STREQ(names[13], "lighting.preview");
    EXPECT_STREQ(names[14], "post");
    EXPECT_STREQ(names[15], "imgui");
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
                ASSERT_EQ(t.gpuSectionMs.size(), 16u);
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

// 256 (and a test failure) for an empty image or images of different sizes
int maxPixelDiff(const TemplatedImage<ucvec4>& a, const TemplatedImage<ucvec4>& b)
{
    if (a.w == 0 || a.h == 0 || a.w != b.w || a.h != b.h)
    {
        ADD_FAILURE() << "maxPixelDiff: images of " << a.w << "x" << a.h << " and " << b.w << "x" << b.h;
        return 256;
    }
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

// takeStatistics returns every frame's statistics in frame order (a bug: when both frame slots completed between two
// reads, the slot of the newer frame was read first and the older frame's statistics were dropped)
TEST_F(VulkanSceneContent, TakeStatisticsInFrameOrder)
{
    CreateScene();
    scene->addMaterial(1, "flat", flatMaterial());
    scene->addMesh(1, "a", boxMesh(vec3(0.3f, 0.3f, 0.3f), vec3(0, 0, 0), {1000}, {ucvec4(255, 0, 0, 255)}), 1);
    camera.position = Vec3(0, 0, 10);
    camera.apply(*scene);
    // Enough visible instances that a frame is still running when the next one is recorded (both slots pending; on
    // lavapipe a frame of these takes ~60 ms, the recording of the next a few ms)
    std::vector<StaticInstance> v;
    for (int i = 0; i < 10000; ++i)
        v.push_back(makeInstance(vec3(-4 + (i % 200) * 0.04f, -2 + (i / 200) * 0.08f, 0), quat::Identity(), 1, 1));
    scene->addInstanceSet(RegionId(), v);
    RenderStatistics st;
    auto noop = [](vk::CommandBuffer) {};
    // An even and an odd number of frames: the newest frame in either slot
    for (int count : {6, 7})
    {
        scene->waitIdle();
        while (scene->takeStatistics(st))
        {
        }
        const uint64_t first = scene->frames() + 1;
        for (int i = 0; i < count; ++i) EXPECT_TRUE(scene->renderFrame(noop, noop, nullptr));
        scene->waitIdle();
        std::vector<uint64_t> frames;
        while (scene->takeStatistics(st))
        {
            frames.push_back(st.frame);
            EXPECT_EQ(st.instancesPerLod[0], 10000u) << st.frame;
        }
        std::vector<uint64_t> expected;
        for (uint64_t f = first; f <= scene->frames(); ++f) expected.push_back(f);
        EXPECT_EQ(frames, expected) << count;
        EXPECT_EQ(scene->statistics().frame, scene->frames());
    }
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
        const Vec3 local    = interpolatePose(a, b, double(alpha)).position;
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

namespace
{
// The pixels that are not the (black) clear color
int coveredPixels(const TemplatedImage<ucvec4>& img)
{
    int n = 0;
    for (int y = 0; y < img.h; ++y)
        for (int x = 0; x < img.w; ++x)
            if (img(y, x)(0) > 0 || img(y, x)(1) > 0 || img(y, x)(2) > 0) n++;
    return n;
}
}  // namespace

// Regression: recordDraw read the mesh pool's current bucket layout and vertex/index buffers, so addMesh and a
// pipeline-class change of setMaterial in the render callback (after the frame's transfer) drew more buckets than the
// commands buffer held (VUID-vkCmdDrawIndexedIndirect-drawCount-00540) and bound the grown vertex buffer before its
// copy. Now the frame draws its transfer's snapshot and the changes take effect in the next frame.
TEST_F(VulkanSceneContent, ContentChangesInRenderCallback)
{
    RenderSceneParameters p;
    // Exactly mesh 1 (4 LODs of 24 vertices and 36 indices): the next mesh grows both buffers
    p.vertexCapacity = 96;
    p.indexCapacity  = 144;
    CreateScene(p);
    scene->addMaterial(1, "opaque", flatMaterial());
    scene->addMaterial(2, "two_sided", flatMaterial(true));
    // Vertex alpha 100: drawn while opaque, discarded once the material is alpha-masked
    const std::vector<float> ends = {10, 20, 30, 100};
    scene->addMesh(1, "box", boxMesh(vec3(0.5f, 0.5f, 0.5f), vec3(0, 0, 0), ends, {ucvec4(220, 180, 40, 100)}), 1);
    camera.position = Vec3(0, 0, 15);
    camera.apply(*scene);
    scene->addInstanceSet(RegionId(), {makeInstance(vec3(-1, 0, 0), quat::Identity(), 1, 1)});
    RenderStatistics st;
    const auto before = SceneFrame(&st);
    EXPECT_EQ(st.instancesPerLod[1], 1u);
    EXPECT_GT(coveredPixels(before), 30);

    // 4 buckets -> 16 (more than the commands buffer's 12), a growth, a class change of mesh 1's material
    MaterialData masked = flatMaterial();
    masked.alphaMask    = true;
    bool changed        = false;
    auto render         = [&](vk::CommandBuffer)
    {
        if (changed) return;
        changed = true;
        for (int k = 0; k < 3; ++k)
            scene->addMesh(MeshId(10 + k), "late" + std::to_string(k),
                           boxMesh(vec3(0.5f, 0.5f, 0.5f), vec3(0, 0, 0), ends, {ucvec4(40, 200, 220, 255)}), 2);
        scene->setMaterial(1, masked);
    };
    TemplatedImage<ucvec4> during;
    scene->requestScreenshot([&](const TemplatedImage<ucvec4>& shot) { during = shot; });
    EXPECT_TRUE(scene->renderFrame([](vk::CommandBuffer) {}, render, nullptr));
    scene->waitIdle();
    st = scene->statistics();
    EXPECT_EQ(st.frame, scene->frames());
    EXPECT_EQ(st.instancesPerLod[1], 1u);
    // The frame of the change is drawn as before
    ASSERT_EQ(during.w, W);
    EXPECT_EQ(maxPixelDiff(during, before), 0);

    // The next frame: mesh 1 alpha-masked (discarded, still counted), a new mesh's instance drawn
    scene->addInstanceSet(RegionId(), {makeInstance(vec3(1, 0, 0), quat::Identity(), 1, 11)});
    const auto after = SceneFrame(&st);
    EXPECT_EQ(st.instancesPerLod[1], 2u);
    EXPECT_EQ(st.nonEmptyBuckets, 2u);
    const Vec2 left = camera.project(Vec3(-1, 0, 0)), right = camera.project(Vec3(1, 0, 0));
    EXPECT_EQ(after(int(left(1)), int(left(0))), ucvec4(0, 0, 0, 255));
    EXPECT_NE(after(int(right(1)), int(right(0))), ucvec4(0, 0, 0, 255));
}

// A pipeline-class change of setMaterial between frames reorders the buckets (mesh 1 moves behind mesh 2): the same
// counts and pixels, then back
TEST_F(VulkanSceneContent, MaterialClassChange)
{
    CreateScene();
    scene->addMaterial(1, "a", flatMaterial());
    scene->addMaterial(2, "b", flatMaterial());
    scene->addMesh(1, "m1", boxMesh(vec3(0.4f, 0.4f, 0.4f), vec3(0, 0, 0), {100}, {ucvec4(255, 0, 0, 255)}), 1);
    scene->addMesh(2, "m2", boxMesh(vec3(0.4f, 0.4f, 0.4f), vec3(0, 0, 0), {15, 100}, {ucvec4(0, 255, 0, 255)}), 2);
    camera.position = Vec3(0, 0, 20);
    camera.apply(*scene);
    scene->addInstanceSet(RegionId(), {makeInstance(vec3(-2, 0, 0), quat::Identity(), 1, 1),
                                       makeInstance(vec3(2, 0, 0), quat::Identity(), 1, 2)});
    RenderStatistics st[3];
    const auto a = SceneFrame(&st[0]);
    scene->setMaterial(1, flatMaterial(true));
    const auto b = SceneFrame(&st[1]);
    scene->setMaterial(2, flatMaterial(true));
    scene->setMaterial(1, flatMaterial(false));
    const auto c = SceneFrame(&st[2]);
    for (int i = 0; i < 3; ++i)
    {
        EXPECT_EQ(st[i].instancesPerLod[0], 1u) << i;
        EXPECT_EQ(st[i].instancesPerLod[1], 1u) << i;
        EXPECT_EQ(st[i].nonEmptyBuckets, 2u) << i;
    }
    EXPECT_GT(coveredPixels(a), 60);
    EXPECT_EQ(maxPixelDiff(a, b), 0);
    EXPECT_EQ(maxPixelDiff(a, c), 0);
}

// usePalette: the vertex color word indexes the palette (an index beyond it is clamped to the last entry); setPalette
// between frames recolors
TEST_F(VulkanSceneContent, PaletteColors)
{
    CreateScene();
    MaterialData pal = flatMaterial();
    pal.usePalette   = true;
    scene->addMaterial(1, "palette", pal);
    scene->setPalette({PaletteEntry{ucvec4(255, 0, 0, 255)}, PaletteEntry{ucvec4(0, 0, 255, 255)}});
    MeshLods first = boxMesh(vec3(0.8f, 0.8f, 0.8f), vec3(0, 0, 0), {100}, {ucvec4(0, 0, 0, 255)});
    MeshLods last  = first;
    for (auto& v : first.lods[0].vertices) v.color = 0;
    for (auto& v : last.lods[0].vertices) v.color = 77;  // beyond the palette: the last entry
    scene->addMesh(1, "first", first, 1);
    scene->addMesh(2, "last", last, 1);
    camera.position = Vec3(0, 0, 10);
    camera.apply(*scene);
    scene->addInstanceSet(RegionId(), {makeInstance(vec3(-2, 0, 0), quat::Identity(), 1, 1),
                                       makeInstance(vec3(2, 0, 0), quat::Identity(), 1, 2)});
    const Vec2 l = camera.project(Vec3(-2, 0, 0.8)), r = camera.project(Vec3(2, 0, 0.8));
    auto img = SceneFrame();
    auto pl = img(int(l(1)), int(l(0))), pr = img(int(r(1)), int(r(0)));
    EXPECT_GT(pl(0), 50);
    EXPECT_EQ(pl(1), 0);
    EXPECT_EQ(pl(2), 0);
    EXPECT_EQ(pr(0), 0);
    EXPECT_EQ(pr(1), 0);
    EXPECT_GT(pr(2), 50);
    scene->setPalette({PaletteEntry{ucvec4(0, 255, 0, 255)}});
    img = SceneFrame();
    pl  = img(int(l(1)), int(l(0)));
    pr  = img(int(r(1)), int(r(0)));
    EXPECT_EQ(pl, pr);
    EXPECT_EQ(pl(0), 0);
    EXPECT_GT(pl(1), 50);
    EXPECT_EQ(pl(2), 0);
}

// alphaMask: vertex alpha * baseColor.a < 0.5 is discarded (counted by the culling, not drawn); without alphaMask the
// alpha is ignored. SmoothTextured too (a bug: its alpha test used only baseColor.a, so a vertex alpha of 100/255
// was drawn)
TEST_F(VulkanSceneContent, AlphaMaskDiscard)
{
    CreateScene();
    MaterialData masked = flatMaterial();
    masked.alphaMask    = true;
    MaterialData faded  = masked;
    faded.baseColor     = vec4(1, 1, 1, 0.4f);
    MaterialData smooth = masked;
    smooth.model        = ShadingModel::SmoothTextured;
    scene->addMaterial(1, "masked", masked);
    scene->addMaterial(2, "opaque", flatMaterial());
    scene->addMaterial(3, "faded", faded);
    scene->addMaterial(4, "smooth", smooth);
    const vec3 half(0.4f, 0.4f, 0.4f);
    scene->addMesh(1, "low", boxMesh(half, vec3(0, 0, 0), {100}, {ucvec4(255, 255, 255, 100)}), 1);
    scene->addMesh(2, "high", boxMesh(half, vec3(0, 0, 0), {100}, {ucvec4(255, 255, 255, 200)}), 1);
    scene->addMesh(3, "low_opaque", boxMesh(half, vec3(0, 0, 0), {100}, {ucvec4(255, 255, 255, 100)}), 2);
    scene->addMesh(4, "faded", boxMesh(half, vec3(0, 0, 0), {100}, {ucvec4(255, 255, 255, 255)}), 3);
    scene->addMesh(5, "smooth_low", boxMesh(half, vec3(0, 0, 0), {100}, {ucvec4(255, 255, 255, 100)}), 4);
    scene->addMesh(6, "smooth_high", boxMesh(half, vec3(0, 0, 0), {100}, {ucvec4(255, 255, 255, 200)}), 4);
    camera.position = Vec3(0, 0, 10);
    camera.apply(*scene);
    const float xs[6] = {-5, -3, -1, 1, 3, 5};
    std::vector<StaticInstance> v;
    for (int i = 0; i < 6; ++i) v.push_back(makeInstance(vec3(xs[i], 0, 0), quat::Identity(), 1, MeshId(i + 1)));
    scene->addInstanceSet(RegionId(), v);
    RenderStatistics st;
    const auto img = SceneFrame(&st);
    EXPECT_EQ(st.instancesPerLod[0], 6u);
    const bool drawn[6] = {false, true, true, false, false, true};
    for (int i = 0; i < 6; ++i)
    {
        const Vec2 p = camera.project(Vec3(xs[i], 0, 0.4));
        EXPECT_EQ(img(int(p(1)), int(p(0))) != ucvec4(0, 0, 0, 255), drawn[i]) << i;
    }
}

// Regression (S19a review): an alpha-masked material with an explicit opaque class (PipelineClass::Opaque or TwoSided)
// was drawn in the opaque bucket, so the depth pre-pass wrote depth where flat.frag discards: the box behind it was
// hidden with the pre-pass only. The masked material goes to the masked class of the same culling
TEST_F(VulkanSceneContent, AlphaMaskedOverrideNotInPrePass)
{
    CreateScene();
    for (auto [c, masked] : {std::pair(PipelineClass::Opaque, PipelineClass::AlphaMasked),
                             std::pair(PipelineClass::TwoSided, PipelineClass::AlphaMaskedTwoSided)})
    {
        MaterialData m  = flatMaterial();
        m.alphaMask     = true;
        m.pipelineClass = uint8_t(c);
        EXPECT_EQ(MeshPool::pipelineClassIndex(m), uint8_t(masked) - 1) << int(c);
    }
    MaterialData opaqueMasked    = flatMaterial();
    opaqueMasked.alphaMask       = true;
    opaqueMasked.pipelineClass   = uint8_t(PipelineClass::Opaque);
    MaterialData twoSidedMasked  = opaqueMasked;
    twoSidedMasked.pipelineClass = uint8_t(PipelineClass::TwoSided);
    scene->addMaterial(1, "flat", flatMaterial());
    scene->addMaterial(2, "opaque_masked", opaqueMasked);
    scene->addMaterial(3, "two_sided_masked", twoSidedMasked);
    // A wall behind two discarded boxes (vertex alpha 100 < 0.5)
    scene->addMesh(1, "wall", boxMesh(vec3(6, 3, 0.2f), vec3(0, 0, 0), {100}, {ucvec4(200, 120, 60, 255)}), 1);
    scene->addMesh(2, "box_opaque", boxMesh(vec3(1, 1, 1), vec3(0, 0, 0), {100}, {ucvec4(255, 255, 255, 100)}), 2);
    scene->addMesh(3, "box_two_sided", boxMesh(vec3(1, 1, 1), vec3(0, 0, 0), {100}, {ucvec4(255, 255, 255, 100)}), 3);
    camera.position = Vec3(0, 0, 10);
    camera.apply(*scene);
    scene->addInstanceSet(RegionId(), {makeInstance(vec3(0, 0, -2), quat::Identity(), 1, 1),
                                       makeInstance(vec3(-2, 0, 2), quat::Identity(), 1, 2),
                                       makeInstance(vec3(2, 0, 2), quat::Identity(), 1, 3)});
    PrototypeSettings ps;
    ps.depthPrePass = false;
    scene->setPrototypeSettings(ps);
    SceneFrame();
    auto without    = SceneFrame();
    ps.depthPrePass = true;
    scene->setPrototypeSettings(ps);
    SceneFrame();
    auto with = SceneFrame();
    EXPECT_EQ(maxPixelDiff(with, without), 0);
    // The wall is visible through both boxes
    for (float x : {-2.0f, 2.0f})
    {
        const Vec2 p = camera.project(Vec3(x, 0, 1));
        EXPECT_NE(with(int(p(1)), int(p(0))), ucvec4(0, 0, 0, 255)) << x;
    }
}

// The storage buffers bound whole stop growing at the limit (maxStorageBufferRange, here a lower test limit of 48 KiB =
// 1536 static instances; 2x growth from 1000 would need 64 KiB) and a set beyond it is a fatal error. The device's
// limit itself is not tested: on lavapipe (128 MiB) it needs 4.2M instances (a 134 MB CPU copy, uploads and buffers).
TEST_F(VulkanSceneContent, StorageBufferLimit)
{
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    RenderSceneParameters p;
    p.staticInstanceCapacity = 16;
    p.dynamicCapacity        = 16;
    p.storageBufferLimit     = 48 * 1024;
    CreateScene(p);
    scene->addMaterial(1, "flat", flatMaterial());
    scene->addMesh(1, "box", boxMesh(vec3(0.02f, 0.02f, 0.02f), vec3(0, 0, 0), {100}, {ucvec4(255, 255, 255, 255)}), 1);
    camera.position = Vec3(0, 0, 10);
    camera.apply(*scene);
    auto grid = [](int n, float y)
    {
        std::vector<StaticInstance> v;
        for (int i = 0; i < n; ++i)
            v.push_back(makeInstance(vec3(-3 + (i % 50) * 0.12f, y + (i / 50) * 0.12f, 0), quat::Identity(), 1, 1));
        return v;
    };
    scene->addInstanceSet(RegionId(), grid(1000, -2));
    RenderStatistics st;
    SceneFrame(&st);
    EXPECT_EQ(st.instancesPerLod[0], 1000u);
    scene->addInstanceSet(RegionId(), grid(536, 0.5f));
    SceneFrame(&st);
    EXPECT_EQ(st.instancesPerLod[0], 1536u);
    EXPECT_DEATH(scene->addInstanceSet(RegionId(), grid(1, 2)),
                 "more than the buffer's limit \\(maxStorageBufferRange");
}

// Regression (S19a review): reserveIds checked next + count, which wrapped around for a huge count (the assert passed
// and the next reservation returned ID 0)
TEST_F(VulkanSceneContent, ReserveIdsRange)
{
    CreateScene();
    EXPECT_EQ(scene->reserveIds(SceneIdKind::Mesh, 1), 1);
    EXPECT_EQ(scene->reserveIds(SceneIdKind::Material, 3), 1);
    EXPECT_EQ(scene->reserveIds(SceneIdKind::Mesh, 2), 2);
#ifdef SAIGA_ASSERTS
    EXPECT_DEATH(scene->reserveIds(SceneIdKind::Mesh, std::numeric_limits<uint32_t>::max()), "is exhausted");
    EXPECT_DEATH(scene->reserveIds(SceneIdKind::Mesh, kGameReservedIds - 3), "is exhausted");
#endif
    // The rest of the range: IDs 4 .. kGameReservedIds - 1
    EXPECT_EQ(scene->reserveIds(SceneIdKind::Mesh, kGameReservedIds - 4), 4);
}

// The count and write passes as a 2D dispatch (work groups limited to 3 per row, 11 chunks: a partial last row) give
// the counts and pixels of the 1D dispatch
TEST_F(VulkanSceneContent, WorkGroupRows)
{
    TemplatedImage<ucvec4> images[2];
    RenderStatistics stats[2];
    for (int run = 0; run < 2; ++run)
    {
        RenderSceneParameters p;
        p.workGroupLimit = run == 0 ? 0 : 3;
        CreateScene(p);
        scene->addMaterial(1, "flat", flatMaterial());
        scene->addMesh(1, "box",
                       boxMesh(vec3(0.05f, 0.05f, 0.05f), vec3(0, 0, 0), {12, 100}, {ucvec4(220, 180, 40, 255)}), 1);
        camera.position = Vec3(0, 0, 10);
        camera.apply(*scene);
        // 2600 instances in two sets: 6 + 5 chunks (the last of each partial)
        std::vector<StaticInstance> a, b;
        for (int i = 0; i < 1500; ++i)
            a.push_back(makeInstance(vec3(-4 + (i % 50) * 0.16f, -3 + (i / 50) * 0.2f, -(i % 7) * 1.0f),
                                     quat::Identity(), 1, 1));
        for (int i = 0; i < 1100; ++i)
            b.push_back(makeInstance(vec3(-4 + (i % 50) * 0.16f, -3 + (i / 50) * 0.2f, -8), quat::Identity(), 1, 1));
        scene->addInstanceSet(RegionId(), a);
        scene->addInstanceSet(RegionId(), b);
        images[run] = SceneFrame(&stats[run]);
    }
    EXPECT_GT(stats[0].instancesPerLod[0], 100u);
    EXPECT_GT(stats[0].instancesPerLod[1], 100u);
    EXPECT_EQ(stats[0].instancesPerLod, stats[1].instancesPerLod);
    EXPECT_EQ(maxPixelDiff(images[0], images[1]), 0);
}

namespace
{
// The camera-relative centre of a dynamic instance's bounding sphere at alpha, as the count pass computes it (in float:
// interpolatePose<float>, the CPU reference of the shader's dynamicPose)
Vec3 dynamicCenter(const ChunkId& chunk, const DynamicPose& prev, const DynamicPose& curr, float alpha,
                   const vec3& sphereCenter, const Vec3& cameraPosition, float* scale)
{
    auto toFloat = [](const DynamicPose& p)
    {
        InstancePose<float> f;
        f.position = p.position.cast<float>();
        f.rotation = quat(float(p.rotation.w()), float(p.rotation.x()), float(p.rotation.y()), float(p.rotation.z()));
        f.scale    = float(p.scale);
        return f;
    };
    const InstancePose<float> pose = interpolatePose(toFloat(prev), toFloat(curr), alpha);
    *scale                         = pose.scale;
    return chunkOrigin(chunk) - cameraPosition + pose.position.cast<double>() +
           rotateD(pose.rotation, sphereCenter.cast<double>() * double(pose.scale));
}

DynamicPose randomPose(std::mt19937& rng, double range)
{
    std::uniform_real_distribution<double> u(-1, 1);
    DynamicPose p;
    p.position = Vec3(u(rng), u(rng), u(rng)) * range;
    p.rotation = Quat(u(rng), u(rng), u(rng), u(rng)).normalized();
    p.scale    = 1.25 + 0.75 * u(rng);
    return p;
}
}  // namespace

// interpolatePose (RenderSceneTypes.h) is the CPU reference of the shaders' dynamic pose: float and double agree, the
// endpoints are the poses, q and -q give the same rotation; a marker 1 m off the mesh origin is drawn where the double
// pose places it (slerp with dot < 0, the nlerp branch near dot = 1, scale and position mixed), and the culling of 400
// random dynamics matches the CPU reference with interpolatePose<float>
TEST_F(VulkanSceneContent, InterpolatePoseMatchesShader)
{
    {
        std::mt19937 rng(7);
        for (int i = 0; i < 50; ++i)
        {
            const DynamicPose a = randomPose(rng, 10), b = randomPose(rng, 10);
            for (double alpha : {0.0, 0.3, 1.0})
            {
                const DynamicPose d = interpolatePose(a, b, alpha);
                InstancePose<float> fa, fb;
                fa.position = a.position.cast<float>();
                fb.position = b.position.cast<float>();
                fa.rotation =
                    quat(float(a.rotation.w()), float(a.rotation.x()), float(a.rotation.y()), float(a.rotation.z()));
                fb.rotation =
                    quat(float(b.rotation.w()), float(b.rotation.x()), float(b.rotation.y()), float(b.rotation.z()));
                fa.scale                    = float(a.scale);
                fb.scale                    = float(b.scale);
                const InstancePose<float> f = interpolatePose(fa, fb, float(alpha));
                EXPECT_LT((f.position.cast<double>() - d.position).norm(), 1e-4);
                EXPECT_LT(
                    std::abs(
                        std::abs(double(f.rotation.w()) * d.rotation.w() + double(f.rotation.x()) * d.rotation.x() +
                                 double(f.rotation.y()) * d.rotation.y() + double(f.rotation.z()) * d.rotation.z()) -
                        1),
                    1e-5);
            }
            const DynamicPose e0 = interpolatePose(a, b, 0.0), e1 = interpolatePose(a, b, 1.0);
            EXPECT_LT((e0.position - a.position).norm(), 1e-12);
            EXPECT_LT((e1.rotation.matrix() - b.rotation.matrix()).norm(), 1e-9);
            DynamicPose nb = b;
            nb.rotation    = Quat(-b.rotation.w(), -b.rotation.x(), -b.rotation.y(), -b.rotation.z());
            EXPECT_LT(
                (interpolatePose(a, nb, 0.4).rotation.matrix() - interpolatePose(a, b, 0.4).rotation.matrix()).norm(),
                1e-9);
        }
    }

    RenderSceneParameters p;
    p.msaaSamples = 4;
    CreateScene(p);
    scene->addMaterial(1, "marker", markerMaterial());
    // A marker 1 m off the mesh origin along x
    scene->addMesh(1, "arm", boxMesh(vec3(0.05f, 0.05f, 0.05f), vec3(1, 0, 0), {100}, {ucvec4(255, 255, 255, 255)}), 1);
    camera.position = Vec3(0, 7, 4);
    camera.rotation = lookRotation(Vec3(0, -1, -0.6));
    camera.apply(*scene);
    struct Case
    {
        Quat q0, q1;
        double s0, s1;
        Vec3 p0, p1;
        float alpha;
    };
    const Vec3 y(0, 1, 0);
    const std::vector<Case> cases = {
        // dot < 0: 0° -> 270° about y (the shorter arc through -45°)
        {Quat::Identity(), Quat(AngleAxis(1.5 * kPi, y)), 1, 1, Vec3(0, 0, 0), Vec3(0, 0, 0), 0.3f},
        // dot > 0.9995: the nlerp branch
        {Quat(AngleAxis(0.5, y)), Quat(AngleAxis(0.52, y)), 1, 1, Vec3(0, 0, 0), Vec3(0, 0, 0), 0.5f},
        // a general rotation with scale and position
        {Quat(AngleAxis(0.4, Vec3(1, 1, 0).normalized())), Quat(AngleAxis(2.0, Vec3(0, 1, 1).normalized())), 1, 2,
         Vec3(-0.5, 0, 0.3), Vec3(0.4, 0.2, -0.6), 0.7f},
    };
    const ChunkId chunk{0, 0, 0};
    auto h = scene->addDynamic(1);
    for (size_t i = 0; i < cases.size(); ++i)
    {
        const Case& c = cases[i];
        DynamicPose a, b;
        a.rotation = c.q0;
        b.rotation = c.q1;
        a.scale    = c.s0;
        b.scale    = c.s1;
        a.position = c.p0;
        b.position = c.p1;
        scene->setDynamic(h, chunk, a, b);
        scene->setInterpolation(c.alpha);
        const DynamicPose pose = interpolatePose(a, b, double(c.alpha));
        const Vec3 marker      = pose.position + pose.rotation.matrix() * Vec3(pose.scale, 0, 0);
        const Vec2 got         = centroid(SceneFrame());
        EXPECT_LT((got - camera.project(marker)).norm(), 0.5)
            << "case " << i << " got " << got.transpose() << " expected " << camera.project(marker).transpose();
    }
    scene->removeDynamic(h);

    // The culling of random dynamics (two chunks) against the CPU reference
    scene->addMesh(2, "box", boxMesh(vec3(0.3f, 0.3f, 0.3f), vec3(1, 0, 0), {10, 20, 35}, {ucvec4(255, 255, 255, 255)}),
                   1);
    camera.position = Vec3(250, 3, 5);
    camera.rotation = lookRotation(Vec3(0.3, -0.1, -1));
    camera.apply(*scene);
    scene->setInterpolation(0.37f);
    std::mt19937 rng(19);
    struct Dyn
    {
        ChunkId chunk;
        DynamicPose a, b;
    };
    std::vector<Dyn> dyns;
    for (int i = 0; i < 400; ++i)
    {
        Dyn d;
        d.chunk = ChunkId{0, i % 2, 0};
        d.a     = randomPose(rng, 25);
        d.b     = randomPose(rng, 25);
        // Around the camera's x (250): chunk 0 near its end, chunk 1 near its start
        const Vec3 shift = Vec3(d.chunk.x == 0 ? 250 : -6, 0, -15);
        d.a.position += shift;
        d.b.position += shift;
        scene->setDynamic(scene->addDynamic(2), d.chunk, d.a, d.b);
        dyns.push_back(d);
    }
    RenderStatistics st;
    SceneFrame(&st);
    Reference ref;
    const auto planes = camera.planes();
    for (const Dyn& d : dyns)
    {
        float scale    = 1;
        const Vec3 c   = dynamicCenter(d.chunk, d.a, d.b, 0.37f, vec3(1, 0, 0), camera.position, &scale);
        const double r = vec3(0.3f, 0.3f, 0.3f).norm() * scale;
        ref.add(c, r, {10, 20, 35}, planes);
    }
    expectCounts(st, ref, "dynamics");
    EXPECT_LT(ref.ambiguous, 10);
}

// Non-decreasing LOD end distances: an end distance equal to the previous one is an empty LOD (never selected, counts
// 0); per mesh and LOD the GPU matches the CPU reference (the tracked statistics). Regression: addMesh asserted
// strictly ascending distances, which the prototype's independently clamped tree LOD settings can violate (S19a).
TEST_F(VulkanSceneContent, EqualDistanceLods)
{
    CreateScene();
    scene->addMaterial(1, "flat", flatMaterial());
    const std::vector<std::vector<float>> ends = {{10, 10, 30}, {10, 25, 25}, {15, 15, 15, 40}, {0, 20}};
    for (size_t m = 0; m < ends.size(); ++m)
        scene->addMesh(MeshId(m + 1), "mesh" + std::to_string(m),
                       boxMesh(vec3(0.2f, 0.2f, 0.2f), vec3(0, 0, 0), ends[m], {ucvec4(255, 255, 255, 255)}), 1);
    scene->trackMeshStatistics({1, 2, 3, 4});
    camera.position = Vec3(0, 0, 0);
    camera.apply(*scene);
    std::vector<StaticInstance> v;
    std::vector<std::array<int, 4>> expected(ends.size(), std::array<int, 4>{});
    Reference total;
    const auto planes = camera.planes();
    for (size_t m = 0; m < ends.size(); ++m)
        for (int i = 0; i < 30; ++i)
        {
            // Distances 2.7 .. 46.2 straight ahead (never within 1e-3 of an end distance)
            const float d = 2.7f + 1.5f * i;
            const vec3 pos(0.3f * float(m) - 0.45f, 0, -d);
            v.push_back(makeInstance(pos, quat::Identity(), 1, MeshId(m + 1)));
            Reference one;
            one.add(pos.cast<double>(), vec3(0.2f, 0.2f, 0.2f).norm(), ends[m], planes);
            total.add(pos.cast<double>(), vec3(0.2f, 0.2f, 0.2f).norm(), ends[m], planes);
            EXPECT_EQ(one.ambiguous, 0);
            for (int l = 0; l < 4; ++l) expected[m][l] += one.perLod[l];
        }
    scene->addInstanceSet(RegionId(), v);
    RenderStatistics st;
    SceneFrame(&st);
    expectCounts(st, total, "equal distances");
    ASSERT_EQ(st.trackedMeshes.size(), ends.size());
    for (size_t m = 0; m < ends.size(); ++m)
    {
        EXPECT_EQ(st.trackedMeshes[m].mesh, MeshId(m + 1));
        for (int l = 0; l < 4; ++l)
            EXPECT_EQ(int(st.trackedMeshes[m].instancesPerLod[l]), expected[m][l]) << "mesh " << m + 1 << " LOD " << l;
    }
    // The empty LODs
    EXPECT_EQ(expected[0][1], 0);
    EXPECT_EQ(expected[1][2], 0);
    EXPECT_EQ(expected[2][1] + expected[2][2], 0);
    EXPECT_EQ(expected[3][0], 0);
    EXPECT_GT(expected[0][2], 0);
    EXPECT_GT(expected[2][3], 0);
}

// trackMeshStatistics: the per-LOD counts of the listed meshes (statics and dynamics), 0 for a registered mesh without
// instances and an unregistered ID, a duplicate twice; each frame's statistics carry the list it was recorded with (a
// change after renderFrame applies to the next frame); a frame without a camera gives zeros; more than maxTrackedMeshes
// is a fatal error
TEST_F(VulkanSceneContent, TrackedMeshStatistics)
{
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    CreateScene();
    scene->addMaterial(1, "flat", flatMaterial());
    scene->addMaterial(2, "two_sided", flatMaterial(true));
    const std::vector<float> ends = {8, 20};
    for (int m = 1; m <= 3; ++m)
        scene->addMesh(MeshId(m), "m" + std::to_string(m),
                       boxMesh(vec3(0.2f, 0.2f, 0.2f), vec3(0, 0, 0), ends, {ucvec4(255, 255, 255, 255)}),
                       MaterialId(m == 2 ? 2 : 1));
    camera.position = Vec3(0, 0, 0);
    camera.apply(*scene);
    // Mesh 1: 3 near + 2 far, mesh 2: 1 near + 4 far, mesh 3: none; a dynamic of mesh 1 (near)
    std::vector<StaticInstance> v;
    auto add = [&](MeshId m, float x, float d) { v.push_back(makeInstance(vec3(x, 0, -d), quat::Identity(), 1, m)); };
    for (float d : {3.f, 5.f, 7.f, 12.f, 15.f}) add(1, -0.5f, d);
    for (float d : {4.f, 10.f, 13.f, 16.f, 18.f}) add(2, 0.5f, d);
    scene->addInstanceSet(RegionId(), v);
    DynamicPose pose;
    pose.position = Vec3(0, 0.5, -6);
    scene->setDynamic(scene->addDynamic(1), ChunkId{0, 0, 0}, pose, pose);

    using Counts                    = std::array<uint32_t, 4>;
    const std::vector<MeshId> listA = {2, 3, 9, 1, 2};
    scene->trackMeshStatistics(listA);
    RenderStatistics st;
    SceneFrame(&st);
    const std::vector<Counts> expectedA = {{1, 4, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {4, 2, 0, 0}, {1, 4, 0, 0}};
    ASSERT_EQ(st.trackedMeshes.size(), listA.size());
    for (size_t i = 0; i < listA.size(); ++i)
    {
        EXPECT_EQ(st.trackedMeshes[i].mesh, listA[i]);
        EXPECT_EQ(st.trackedMeshes[i].instancesPerLod, expectedA[i]) << i;
    }
    EXPECT_EQ(st.instancesPerLod[0], 5u);
    EXPECT_EQ(st.instancesPerLod[1], 6u);

    // A list change after renderFrame: that frame still carries list A, the next one the new list
    EXPECT_TRUE(scene->renderFrame([](vk::CommandBuffer) {}, [](vk::CommandBuffer) {}, nullptr));
    const uint64_t frameA = scene->frames();
    scene->trackMeshStatistics({1});
    scene->waitIdle();
    st = scene->statistics();
    EXPECT_EQ(st.frame, frameA);
    ASSERT_EQ(st.trackedMeshes.size(), listA.size());
    EXPECT_EQ(st.trackedMeshes[0].mesh, MeshId(2));
    SceneFrame(&st);
    ASSERT_EQ(st.trackedMeshes.size(), 1u);
    EXPECT_EQ(st.trackedMeshes[0].mesh, MeshId(1));
    EXPECT_EQ(st.trackedMeshes[0].instancesPerLod, (Counts{4, 2, 0, 0}));

    // An empty list
    scene->trackMeshStatistics({});
    SceneFrame(&st);
    EXPECT_TRUE(st.trackedMeshes.empty());

    // Without a camera (a new scene): the list with zeros
    CreateScene();
    scene->addMaterial(1, "flat", flatMaterial());
    scene->addMesh(1, "m", boxMesh(vec3(0.2f, 0.2f, 0.2f), vec3(0, 0, 0), ends, {ucvec4(255, 255, 255, 255)}), 1);
    scene->addInstanceSet(RegionId(), {makeInstance(vec3(0, 0, -3), quat::Identity(), 1, 1)});
    scene->trackMeshStatistics({1});
    SceneFrame(&st);
    ASSERT_EQ(st.trackedMeshes.size(), 1u);
    EXPECT_EQ(st.trackedMeshes[0].instancesPerLod, (Counts{0, 0, 0, 0}));

    EXPECT_DEATH(scene->trackMeshStatistics(std::vector<MeshId>(maxTrackedMeshes + 1, 1)), "at most maxTrackedMeshes");
}

// S19a: the depth pre-pass of the opaque content (the prototype terrain and water, an opaque instance: static and
// dynamic at alpha 0.5) gives the same image as the scene pass alone (depth test and write): the vertex shaders write
// invariant positions, the scene pass draws them with EQUAL. MSAA with the SAMPLE_ZERO depth resolve and 1 sample
TEST_F(VulkanSceneContent, DepthPrePassMatches)
{
    for (int samples : {4, 1})
    {
        RenderSceneParameters p;
        p.msaaSamples = samples;
        CreateScene(p);
        // S20: test tiles of an integer hook on the GPU terrain set (with the environment's light)
        PrototypeSettings ps;
        ps.grass = false;
        scene->setFoliageTexture(1);
        scene->setPrototypeSettings(ps);
        const TerrainLayout layout = TerrainTestHooks::testLayout(16, 2);
        const auto source          = TerrainHeightSource::fromHook(TerrainTestHooks::slopeRidgeHook, 1);
        scene->addMaterial(1, "flat", flatMaterial());
        scene->addMesh(1, "box", boxMesh(vec3(3, 3, 3), vec3(0, 0, 0), {1000}, {ucvec4(200, 120, 60, 255)}), 1);
        // A box half in the ground (the terrain and the box test against each other) and a moving one
        const float x = 128, z = 120;
        const float h = float(TerrainTestHooks::slopeRidgeHook(1, 128, 120) * kHeightUnit);
        scene->addInstanceSet(RegionId{0, 0, 0, 0},
                              {makeInstance(vec3(x, h, z), axisAngle(vec3(0, 1, 0), 0.4f), 1, 1)});
        DynamicPose a, b;
        a.position = Vec3(x + 8, h + 1, z - 4);
        b.position = Vec3(x + 14, h - 1, z + 2);
        b.rotation = Quat(AngleAxis(0.9, Vec3(0, 1, 0)));
        scene->setDynamic(scene->addDynamic(1), ChunkId{0, 0, 0}, a, b);
        scene->setInterpolation(0.5f);
        camera.position = Vec3(x - 30, h + 12, z + 30);
        camera.rotation = lookRotation(Vec3(30, -10, -30));
        camera.apply(*scene);
        TerrainTestHooks::fillTestTerrain(*scene, layout, source, 0, camera.position, false);

        ps.depthPrePass = true;
        scene->setPrototypeSettings(ps);
        SceneFrame();
        RenderStatistics st;
        auto with = SceneFrame(&st);
        EXPECT_EQ(st.instancesPerLod[0], 2u) << samples;
        EXPECT_GT(st.terrainNodes, 0u);
        ps.depthPrePass = false;
        scene->setPrototypeSettings(ps);
        SceneFrame();
        auto without = SceneFrame();
        EXPECT_EQ(maxPixelDiff(with, without), 0) << samples << " samples";
        // The boxes are drawn (their color, not the gray clay terrain)
        int colored = 0;
        for (int y = 0; y < with.h; ++y)
            for (int xx = 0; xx < with.w; ++xx) colored += with(y, xx)(0) > with(y, xx)(2) + 20;
        EXPECT_GT(colored, 50) << samples;
    }
}

//==================================================================================================
// Spaces (S21, docs/engine_design.md §10.1, §12.4): the camera space's render settings

namespace
{
// The camera's space 1 enclosed with a bright background (so the tone-mapped value is well above 0)
SpaceRenderSettings enclosedSettings()
{
    SpaceRenderSettings s;
    s.openWorld  = false;
    s.background = vec3(0.3f, 0.1f, 0.05f);
    return s;
}
}  // namespace

// Pipeline reuse (VulkanBase::reusePipelines, on for the tests' reused headless devices): a destroyed scene returns its
// pipelines to the device's pool, a scene with the same parameters takes every one of them and renders the same image
// (the pooled pipelines bound with the new scene's own, identically defined layouts; validation clean); a scene with
// another sample count takes other pipelines (the formats and samples are part of the key)
TEST_F(VulkanSceneContent, PipelinesReusedAcrossScenes)
{
    auto build = [&](int samples, VulkanBase::PipelineReuseStats& before, VulkanBase::PipelineReuseStats& after)
    {
        RenderSceneParameters p;
        p.msaaSamples = samples;
        // The stats of the pooled device before this scene's pipelines: from the previous scene (none at first)
        if (scene) before = scene->base().pipelineReuseStats();
        CreateScene(p);
        after = scene->base().pipelineReuseStats();
        scene->addMaterial(1, "flat", flatMaterial());
        scene->addMesh(1, "box", boxMesh(vec3(1, 1, 1), vec3(0, 0, 0), {1000}, {ucvec4(200, 120, 60, 255)}), 1);
        scene->addInstanceSet(RegionId(), {makeInstance(vec3(0, 0, -5), axisAngle(vec3(0, 1, 0), 0.4f), 1, 1)});
        camera.position = Vec3(0, 0, 0);
        camera.apply(*scene);
        return SceneFrame();
    };
    VulkanBase::PipelineReuseStats b0, a0, b1, a1, b2, a2;
    const auto first = build(4, b0, a0);
    if (!scene->base().reusePipelines) GTEST_SKIP() << "pipeline reuse is off (SAIGA_TEST_NEW_DEVICES=1)";
    const auto second = build(4, b1, a1);
    // Every pipeline of the second scene came from the pool (the same device: the pool holds one device per key)
    EXPECT_GT(a1.hits - b1.hits, 10u);
    EXPECT_EQ(a1.misses, b1.misses);
    EXPECT_EQ(maxPixelDiff(first, second), 0);
    int colored = 0;
    for (int y = 0; y < second.h; ++y)
        for (int x = 0; x < second.w; ++x) colored += second(y, x)(0) > second(y, x)(2) + 20;
    EXPECT_GT(colored, 50);
    // One sample: other keys for the multisampled pipelines (a pipeline of the 4x pass in the 1x pass would be a
    // validation error, which fails the test); the box is drawn
    const auto single = build(1, b2, a2);
    EXPECT_GT(a2.hits + a2.misses, b2.hits + b2.misses);
    colored = 0;
    for (int y = 0; y < single.h; ++y)
        for (int x = 0; x < single.w; ++x) colored += single(y, x)(0) > single(y, x)(2) + 20;
    EXPECT_GT(colored, 50);
}

class VulkanSceneSpaces : public VulkanSceneContent
{
};

// An enclosed camera space: no terrain, grass, water or sky (a terrain set of that space is ignored), the background
// as the clear color through the tone map; its dynamic instances drawn, those of space 0 not; without an environment
// the instances are lit by the enclosed studio light (S36: prototypeStudio() without fog; S21's kEnclosedLighting)
TEST_F(VulkanSceneSpaces, EnclosedSpace)
{
    RenderSceneParameters p;
    p.toneMap     = ToneMap::Aces;
    p.msaaSamples = 4;
    CreateScene(p);
    scene->setFoliageTexture(1);
    scene->setSpaceSettings(1, enclosedSettings());
    EXPECT_FALSE(scene->spaceSettings(1).openWorld);
    EXPECT_TRUE(scene->spaceSettings(0).openWorld);
    EXPECT_TRUE(scene->spaceSettings(7).openWorld);  // no settings: open world
    scene->addMaterial(1, "flat", flatMaterial());
    scene->addMesh(1, "box", boxMesh(vec3(1, 1, 1), vec3(0, 0, 0), {1000}, {ucvec4(200, 120, 60, 255)}), 1);
    DynamicPose inside, outside;
    inside.position  = Vec3(100, 2, 96);
    outside.position = Vec3(104, 2, 96);
    scene->setDynamic(scene->addDynamic(1), ChunkId{1, 0, 0}, inside, inside);
    scene->setDynamic(scene->addDynamic(1), ChunkId{0, 0, 0}, outside, outside);
    camera.space    = 1;
    camera.position = Vec3(102, 2, 106);
    camera.rotation = lookRotation(Vec3(0, 0, -1));
    camera.apply(*scene);
    // A terrain set of the camera's space with water (an enclosed space never gets one; ignored if it did)
    const TerrainLayout layout = TerrainTestHooks::testLayout(16, 2);
    TerrainTestHooks::fillTestTerrain(*scene, layout,
                                      TerrainHeightSource::fromHook(TerrainTestHooks::slopeRidgeHook, 1), 4096,
                                      camera.position, true, 1);
    SceneFrame();
    RenderStatistics st;
    const auto enclosed = SceneFrame(&st);
    EXPECT_EQ(st.terrainNodes, 0u);
    EXPECT_EQ(scene->terrainPassStats().nodes, 0);
    EXPECT_FALSE(scene->terrainPassStats().water);
    EXPECT_EQ(scene->grassPass().stats().blocks, 0);
    EXPECT_EQ(st.instancesPerLod[0], 1u);  // the box of space 1 only
    // The background (a corner): the space's linear background through ACES and the sRGB output
    const vec3 b = enclosedSettings().background;
    for (int c = 0; c < 3; ++c)
    {
        EXPECT_NEAR(int(enclosed(2, 2)(c)), acesPixel(b[c]), 2) << c;
        EXPECT_NEAR(int(enclosed(H - 3, W - 3)(c)), acesPixel(b[c]), 2) << c;
    }
    // The box of space 1 is drawn, the one of space 0 is not
    const Vec2 in = camera.project(inside.position + Vec3(0, 0, 1)), out = camera.project(outside.position);
    const ucvec4 lit = enclosed(int(in(1)), int(in(0)));
    EXPECT_GT(std::abs(int(lit(0)) - acesPixel(b[0])), 10);
    for (int c = 0; c < 3; ++c) EXPECT_NEAR(int(enclosed(int(out(1)), int(out(0)))(c)), acesPixel(b[c]), 2);

    // The reference: the same box in an open-world space without terrain, lit by the enclosed studio light (the
    // prototype's without fog; the sky off)
    EXPECT_FALSE(scene->studioEnvironment(true).fog);
    PrototypeSettings reference;
    reference.sky = false;
    scene->setPrototypeSettings(reference);
    scene->setStudioEnvironment(scene->studioEnvironment(true));
    scene->setDynamic(scene->addDynamic(1), ChunkId{2, 0, 0}, inside, inside);
    camera.space = 2;
    camera.apply(*scene);
    SceneFrame();
    const auto open = SceneFrame(&st);
    EXPECT_EQ(st.instancesPerLod[0], 1u);
    const ucvec4 ref = open(int(in(1)), int(in(0)));
    for (int c = 0; c < 3; ++c) EXPECT_NEAR(int(lit(c)), int(ref(c)), 1) << c;
    // Another light (a low, bright sun with more ambient) lights it differently (the comparison above means something)
    StudioEnvironment game = prototypeStudio();
    game.sunDir            = vec3(0.3f, 0.17f, 0.94f).normalized();
    game.sunIntensity      = 2.5f;
    game.ambient           = 0.6f;
    scene->setStudioEnvironment(game);
    SceneFrame();
    const auto gameLit = SceneFrame();
    const ucvec4 g     = gameLit(int(in(1)), int(in(0)));
    EXPECT_GT(std::abs(int(g(0)) - int(lit(0))) + std::abs(int(g(1)) - int(lit(1))), 4);
}

// An open-world frame, an enclosed one, the open-world frame again: the first and the last are equal within the golden
// metrics (§17.6)
TEST_F(VulkanSceneSpaces, SwitchBack)
{
    RenderSceneParameters p;
    p.toneMap = ToneMap::Aces;
    CreateScene(p);
    scene->setFoliageTexture(1);
    scene->setSpaceSettings(1, enclosedSettings());
    scene->addMaterial(1, "flat", flatMaterial());
    scene->addMesh(1, "box", boxMesh(vec3(1, 1, 1), vec3(0, 0, 0), {1000}, {ucvec4(200, 120, 60, 255)}), 1);
    const float h = float(TerrainTestHooks::slopeRidgeHook(1, 128, 120) * kHeightUnit);
    scene->addInstanceSet(RegionId{0, 0, 0, 0}, {makeInstance(vec3(128, h + 1, 110), quat::Identity(), 1, 1)});
    camera.position = Vec3(128, h + 6, 130);
    camera.rotation = lookRotation(Vec3(0, -0.2, -1));
    camera.apply(*scene);
    const TerrainLayout layout = TerrainTestHooks::testLayout(16, 2);
    TerrainTestHooks::fillTestTerrain(*scene, layout,
                                      TerrainHeightSource::fromHook(TerrainTestHooks::slopeRidgeHook, 1), 0,
                                      camera.position, true, 0);
    SceneFrame();
    RenderStatistics st;
    const auto first = SceneFrame(&st);
    EXPECT_GT(st.terrainNodes, 0u);
    TestCamera inside = camera;
    inside.space      = 1;
    inside.apply(*scene);
    SceneFrame();
    const auto enclosed = SceneFrame(&st);
    EXPECT_EQ(st.terrainNodes, 0u);
    EXPECT_GT(goldenMetrics(first, enclosed).meanAbs, 5);
    camera.apply(*scene);
    SceneFrame();
    const auto last = SceneFrame(&st);
    EXPECT_GT(st.terrainNodes, 0u);
    const GoldenMetrics m = goldenMetrics(first, last);
    EXPECT_TRUE(m.pass) << "mean " << m.meanAbs << ", large " << m.largeFraction << ", max " << m.maxDelta;
}

// A space without settings (the vulkan samples and tests that never call setSpaceSettings) is drawn as open world
TEST_F(VulkanSceneSpaces, NoSettingsMeansOpenWorld)
{
    CreateScene();
    scene->setFoliageTexture(1);
    scene->setSpaceSettings(1, enclosedSettings());
    // Regression (PR review, S21): the settings of another space as the argument (an element of the resized vector)
    scene->setSpaceSettings(40, scene->spaceSettings(1));
    EXPECT_FALSE(scene->spaceSettings(40).openWorld);
    EXPECT_TRUE(scene->spaceSettings(3).openWorld);
    camera.space    = 3;
    camera.position = Vec3(128, 40, 130);
    camera.rotation = lookRotation(Vec3(0, -0.5, -1));
    camera.apply(*scene);
    TerrainTestHooks::fillTestTerrain(*scene, TerrainTestHooks::testLayout(16, 2),
                                      TerrainHeightSource::fromHook(TerrainTestHooks::slopeRidgeHook, 1), 0,
                                      camera.position, false, 3);
    SceneFrame();
    RenderStatistics st;
    SceneFrame(&st);
    EXPECT_GT(st.terrainNodes, 0u);
    EXPECT_TRUE(scene->terrainPassStats().water);
}
