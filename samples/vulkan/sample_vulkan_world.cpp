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
 */

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/util/mcp/McpServer.h"
#include "saiga/vulkan/window/GLFWSample.h"
#include "saiga/vulkan/world/WorldMcp.h"
#include "saiga/vulkan/world/WorldRenderer.h"

#include <cstdlib>
#include <iostream>

using namespace Saiga;

static Vulkan::ForwardRenderingParameters rendererParameters()
{
    Vulkan::ForwardRenderingParameters p;
    p.msaaSamples = 4;
    p.reversedZ   = true;
    return p;
}

class WorldSample : public VulkanExampleBase
{
   public:
    WorldSample() : VulkanExampleBase(rendererParameters()), mcp("saiga-world", "1.0", Vulkan::worldMcpInstructions())
    {
        Vulkan::WorldParameters params;
        if (const char* trees = std::getenv("SAIGA_WORLD_TREES")) params.treeCount = std::atoi(trees);
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

        int port = 8765;
        if (const char* p = std::getenv("SAIGA_MCP_PORT")) port = std::atoi(p);
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
        world.transfer(cmd, camera.view, camera.getPosition(), window->getAspectRatio());
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

   private:
    Vulkan::WorldRenderer world;
    McpServer mcp;
    bool walkMode   = false;
    float eyeHeight = 1.8f;
};

int main(const int argc, const char* argv[])
{
    initSaigaSample();
    WorldSample sample;
    MainLoopParameters mainLoop;
    // As fast as the swap chain allows
    mainLoop.framesPerSecond = 0;
    sample.run(mainLoop);
    return 0;
}
