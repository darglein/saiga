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
#include "saiga/vulkan/scene/RenderScene.h"
#include "saiga/vulkan/window/HeadlessWindow.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>

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
    ASSERT_EQ(names.size(), 5u);
    EXPECT_STREQ(names[0], "transfer");
    EXPECT_STREQ(names[1], "scene");
    EXPECT_STREQ(names[2], "resolve");
    EXPECT_STREQ(names[3], "post");
    EXPECT_STREQ(names[4], "imgui");
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
                ASSERT_EQ(t.gpuSectionMs.size(), 5u);
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
