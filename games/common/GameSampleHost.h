/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/imgui/imgui.h"
#include "saiga/vulkan/window/GLFWSample.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>

namespace Saiga
{
namespace Game
{
/**
 * The shared window and loop wiring of the tech demo executables (samples/vulkan/sample_vulkan_<game>.cpp): a GLFW
 * window with the forward renderer (MSAA 4, reversed Z), the game stepped at a fixed 60 Hz in update(), rendered every
 * frame with the interpolation factor, R respawns, the game's ImGui window. A sample derives, initializes the game in
 * its constructor and maps the input in stepGame(). Header only (needs GLFW, so only the samples include it).
 * Transitional (docs/engine_design.md §18.4): replaced by EngineApp (S13).
 */
template <class GameT>
class GameSampleHost : public VulkanExampleBase
{
   public:
    static constexpr double stepDt = 1.0 / 60.0;

    explicit GameSampleHost(const vec4& clearColor = Vulkan::ForwardRenderingParameters().clearColor)
        : VulkanExampleBase(rendererParameters(clearColor))
    {
    }
    ~GameSampleHost() override
    {
        renderer->waitIdle();
        game.destroy();
    }

    // Runs the fixed-step main loop until the window closes
    void runGame()
    {
        MainLoopParameters mainLoop;
        mainLoop.updatesPerSecond = 60;
        mainLoop.framesPerSecond  = 0;
        run(mainLoop);
    }

    static bool hasFlag(int argc, char** argv, const char* flag)
    {
        for (int i = 1; i < argc; ++i)
            if (std::strcmp(argv[i], flag) == 0) return true;
        return false;
    }
    // The tree count of the world prototype (SAIGA_WORLD_TREES overrides it, e.g. for slow GPUs)
    static int worldTreeCount(int fallback = 200000)
    {
        const char* t = std::getenv("SAIGA_WORLD_TREES");
        return t ? std::atoi(t) : fallback;
    }

   protected:
    // One fixed step: read the input (unless ImGui has the keyboard) and step the game
    virtual void stepGame(double dt, bool keyboardFree) = 0;

    GameT game;

   private:
    static Vulkan::ForwardRenderingParameters rendererParameters(const vec4& clearColor)
    {
        Vulkan::ForwardRenderingParameters p;
        p.msaaSamples = 4;
        p.reversedZ   = true;
        p.clearColor  = clearColor;
        return p;
    }

    void update(float) override { stepGame(stepDt, !ImGui::captureKeyboard()); }
    void interpolate(float, float a) override { alpha = a; }
    void transfer(vk::CommandBuffer cmd) override
    {
        auto now  = std::chrono::steady_clock::now();
        double dt = started ? std::chrono::duration<double>(now - last).count() : stepDt;
        last      = now;
        started   = true;
        game.transfer(cmd, window->getAspectRatio(), std::min(dt, 0.1), alpha);
    }
    void render(vk::CommandBuffer cmd) override { game.render(cmd); }
    void renderGUI() override
    {
        game.imgui();
        window->renderImGui();
    }
    void keyPressed(int key, int scancode, int mods) override
    {
        if (key == GLFW_KEY_R && !ImGui::captureKeyboard()) game.respawn();
        VulkanExampleBase::keyPressed(key, scancode, mods);
    }

    double alpha = 0;
    std::chrono::steady_clock::time_point last;
    bool started = false;
};

}  // namespace Game
}  // namespace Saiga
