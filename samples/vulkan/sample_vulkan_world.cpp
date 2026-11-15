/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

/**
 * A large procedural world (8 x 8 km terrain, 200k trees, grass around the camera) with the WorldRenderer of the
 * Vulkan module, rendered in a uniform gray to judge the geometry.
 *
 * Controls: WASD + mouse (left or middle button) to fly, shift = fast, space = up, F toggles the walk mode, G the GUI.
 *
 * The sample runs an MCP server (http://127.0.0.1:8765/mcp, port from SAIGA_MCP_PORT, 0 disables it), so an agent
 * can take screenshots and control the camera and the settings:
 *   claude mcp add --transport http saiga-world http://127.0.0.1:8765/mcp
 *
 * Benchmarks (docs/engine_design.md §17.7): sample_vulkan_world --benchmark <flyover|forest> flies a fixed camera path
 * at 1920 x 1080 (validation, vsync, GUI and MCP off, frozen time, the default world: SAIGA_WORLD_TREES is ignored),
 * writes benchmark_<name>.json to the working directory and exits with 0, or with 1 if the benchmark did not finish or
 * the result could not be written (an old result file is deleted at the start).
 */

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/util/mcp/McpServer.h"
#include "saiga/vulkan/window/GLFWSample.h"
#include "saiga/vulkan/world/WorldBenchmark.h"
#include "saiga/vulkan/world/WorldMcp.h"
#include "saiga/vulkan/world/WorldRenderer.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>

using namespace Saiga;

static Vulkan::ForwardRenderingParameters rendererParameters()
{
    Vulkan::ForwardRenderingParameters p;
    p.msaaSamples = 4;
    p.reversedZ   = true;
    return p;
}

// The window from config.ini; a benchmark uses 1920 x 1080 without vsync
static WindowParameters windowParameters(bool benchmark)
{
    WindowParameters w;
    w.fromConfigFile("config.ini");
    if (benchmark)
    {
        w.width  = 1920;
        w.height = 1080;
        w.mode   = WindowParameters::Mode::Windowed;
        w.vsync  = false;
    }
    return w;
}

// The Vulkan parameters from config.ini; a benchmark turns validation, vsync, ImGui and defragmentation off, so the
// result does not depend on the user's config.ini
static Vulkan::VulkanParameters vulkanParameters(bool benchmark)
{
    Vulkan::VulkanParameters v;
    v.fromConfigFile("config.ini");
    if (benchmark)
    {
        v.enableValidationLayer = false;
        v.vsync                 = false;
        v.enableImgui           = false;
        v.enableDefragmentation = false;
    }
    return v;
}

class WorldSample : public VulkanExampleBase
{
   public:
    // benchmark: the name of a WorldBenchmark, empty for the interactive sample
    explicit WorldSample(const std::string& benchmark)
        : VulkanExampleBase(windowParameters(!benchmark.empty()), vulkanParameters(!benchmark.empty()),
                            rendererParameters()),
          mcp("saiga-world", "1.0", Vulkan::worldMcpInstructions())
    {
        Vulkan::WorldParameters params;
        // A benchmark always uses the default world (its result is compared with the baseline)
        if (const char* trees = std::getenv("SAIGA_WORLD_TREES"); trees && benchmark.empty())
            params.treeCount = std::atoi(trees);
        world.init(*renderer, params);

        // First person controls with a fixed up vector on both mouse buttons
        camera.mode0             = CameraControlMode::ROTATE_FIRST_PERSON_FIX_UP_VECTOR;
        camera.mode1             = CameraControlMode::ROTATE_FIRST_PERSON_FIX_UP_VECTOR;
        camera.movementSpeed     = 25;
        camera.movementSpeedFast = 250;

        // Start on a hill: next to the highest point, looking over the world center
        auto& hf    = world.heightField();
        vec3 peak   = hf.highestPoint();
        vec3 center = vec3(hf.worldSize() * 0.5f, 0, hf.worldSize() * 0.5f);
        vec3 toPeak = vec3(peak(0) - center(0), 0, peak(2) - center(2));
        vec3 eye    = peak + (toPeak.norm() > 1 ? toPeak.normalized() * 150.0f : vec3(150, 0, 0));
        eye(1)      = std::max(hf.height(eye(0), eye(2)), peak(1) - 40) + 60;
        center(1)   = hf.height(center(0), center(2));
        camera.setView(eye, center, vec3(0, 1, 0));

        benchmarkName = benchmark;
        if (!benchmark.empty())
        {
            // Frozen time, no GUI, no MCP: only the world is measured
            world.settings.freezeTime     = true;
            renderer->should_render_imgui = false;
            std::string gpu               = renderer->base().physicalDevice.getProperties().deviceName.data();
            bench                         = std::make_unique<Vulkan::WorldBenchmark>(benchmark, world, gpu);
            // No stale result: the file exists after the run only if this run wrote it
            std::remove(resultFile().c_str());
            // The automatic screenshot of SAIGA_SCREENSHOT would close the window at frame 60
            if (renderer->disableAutoScreenshot())
                std::cout << "Benchmark: SAIGA_SCREENSHOT is ignored in benchmark mode" << std::endl;
            std::cout << "Benchmark " << benchmark << " on " << gpu << std::endl;
        }

        int port = 8765;
        if (const char* p = std::getenv("SAIGA_MCP_PORT")) port = std::atoi(p);
        if (!benchmark.empty()) port = 0;
        if (port > 0)
        {
            Vulkan::WorldMcpContext context;
            context.renderer  = renderer.get();
            context.world     = &world;
            context.camera    = &camera;
            context.walkMode  = &walkMode;
            context.eyeHeight = &eyeHeight;
            // Before the tools are registered: measure_performance answers within the timeout
            mcp.toolTimeoutSeconds = 120;
            Vulkan::registerWorldMcpTools(mcp, context);
            if (mcp.start(port))
                std::cout << "MCP server: http://127.0.0.1:" << mcp.port() << "/mcp" << std::endl;
            else
                std::cout << "MCP server: could not listen on port " << port << std::endl;
        }
    }

    ~WorldSample() override
    {
        mcp.stop();
        renderer->waitIdle();
        world.destroy();
    }

    void update(float dt) override
    {
        VulkanExampleBase::update(dt);
        if (walkMode)
        {
            vec3 p          = camera.getPosition();
            p(1)            = world.heightField().height(p(0), p(2)) + eyeHeight;
            camera.position = make_vec4(p, 1);
            camera.calculateModel();
            camera.updateFromModel();
        }
        mcp.poll();
    }

    void transfer(vk::CommandBuffer cmd) override
    {
        if (bench)
        {
            auto pose = bench->currentPose();
            camera.setView(pose.eye, pose.target, vec3(0, 1, 0));
            // The size that is actually rendered: the swap chain extent (it may differ from the requested
            // 1920 x 1080, e.g. clamped by the surface). Only the measured frames count (a window manager may resize
            // the window during the warm-up); a run whose size changed while measuring is not comparable.
            if (bench->measuring())
            {
                ivec2 size = renderer->surfaceSize();
                int w = size(0), h = size(1);
                if (bench->width != 0 && (w != bench->width || h != bench->height)) sizeChanged = true;
                bench->width  = w;
                bench->height = h;
            }
        }
        world.transfer(cmd, camera.view, camera.getPosition(), window->getAspectRatio());
        if (bench && bench->finished() && !benchmarkDone)
        {
            std::string file = resultFile();
            if (bench->width != 1920 || bench->height != 1080)
                std::cout << "Benchmark warning: rendered at " << bench->width << " x " << bench->height
                          << " instead of 1920 x 1080 (not comparable with the baseline)" << std::endl;
            if (sizeChanged) std::cout << "Benchmark error: the rendered size changed during the run" << std::endl;
            json11::Json result = bench->result();
            if (result["gpu_ms"].is_null())
                std::cout << "Benchmark warning: no GPU times (timestamps not supported)" << std::endl;
            bool written = Vulkan::WorldBenchmark::writeResult(file, result);
            if (written)
                std::cout << "Benchmark result: " << file << "\n" << result.dump() << std::endl;
            else
                std::cout << "Benchmark error: could not write " << file << std::endl;
            benchmarkOk   = written && !sizeChanged;
            benchmarkDone = true;
            window->close();
        }
    }

    void render(vk::CommandBuffer cmd) override { world.render(cmd); }

    void renderGUI() override
    {
        world.imgui();
        ImGui::Begin("World");
        ImGui::Separator();
        ImGui::Checkbox("walk mode (F)", &walkMode);
        ImGui::SliderFloat("eye height", &eyeHeight, 0.5f, 50.0f);
        ImGui::SliderFloat("speed", &camera.movementSpeed, 1, 200);
        ImGui::SliderFloat("speed fast", &camera.movementSpeedFast, 10, 2000);
        vec3 p = camera.getPosition();
        ImGui::Text("camera %.1f %.1f %.1f (%.1f m above ground)", p(0), p(1), p(2),
                    p(1) - world.heightField().height(p(0), p(2)));
        if (mcp.running()) ImGui::Text("MCP: http://127.0.0.1:%d/mcp", mcp.port());
        ImGui::End();
        VulkanExampleBase::renderGUI();
    }

    void keyPressed(int key, int scancode, int mods) override
    {
        if (!ImGui::captureKeyboard() && key == GLFW_KEY_F) walkMode = !walkMode;
        VulkanExampleBase::keyPressed(key, scancode, mods);
    }

    // The benchmark finished, its result was written and the rendered size did not change (exit code 0)
    bool benchmarkSucceeded() const { return benchmarkOk; }
    // The benchmark finished (the result may still have failed: write error or size change)
    bool benchmarkFinished() const { return benchmarkDone; }

   private:
    std::string resultFile() const { return "benchmark_" + benchmarkName + ".json"; }

    std::string benchmarkName;
    Vulkan::WorldRenderer world;
    McpServer mcp;
    std::unique_ptr<Vulkan::WorldBenchmark> bench;
    bool benchmarkDone = false, benchmarkOk = false, sizeChanged = false;
    bool walkMode   = false;
    float eyeHeight = 1.8f;
};

int main(const int argc, const char* argv[])
{
    std::string benchmark;
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--benchmark") == 0)
        {
            if (i + 1 >= argc)
            {
                std::cout << "--benchmark needs a name" << std::endl;
                return 1;
            }
            benchmark = argv[++i];
        }
    }
    if (!benchmark.empty() && !Vulkan::WorldBenchmark::known(benchmark))
    {
        std::cout << "Unknown benchmark '" << benchmark << "'. Known:";
        for (auto& n : Vulkan::WorldBenchmark::names()) std::cout << " " << n;
        std::cout << std::endl;
        return 1;
    }

    initSaigaSample();
    WorldSample sample(benchmark);
    MainLoopParameters mainLoop;
    // As fast as the swap chain allows
    mainLoop.framesPerSecond = 0;
    sample.run(mainLoop);
    // A benchmark run fails if it was closed early or the result could not be written
    if (!benchmark.empty() && !sample.benchmarkSucceeded())
    {
        // The reason of a finished but failed run was printed above
        if (!sample.benchmarkFinished()) std::cout << "Benchmark " << benchmark << " did not finish" << std::endl;
        return 1;
    }
    return 0;
}
