/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Jolt Physics in a Vulkan window (built with -DSAIGA_WITH_JOLT=ON): a pyramid and a wall of boxes. A heavy ball is
// fired into the pyramid at the start.
//
// Controls (flying first-person camera): drag with the left or right mouse button to look around, WASD to move,
// SPACE / CTRL up and down, SHIFT fast. F or the middle mouse button fires a ball, R resets the scene.
// The simulation runs at the fixed update rate of the main loop (60 Hz) on all cores.

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/model/model_from_shape.h"
#include "saiga/core/util/color.h"
#include "saiga/vulkan/renderModules/AssetRenderer.h"
#include "saiga/vulkan/window/GLFWSample.h"

#include <Jolt/Jolt.h>

#include <Jolt/Core/Factory.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/RegisterTypes.h>

#include <algorithm>
#include <memory>
#include <thread>
#include <vector>

using namespace Saiga;

namespace Layers
{
static constexpr JPH::ObjectLayer STATIC  = 0;
static constexpr JPH::ObjectLayer DYNAMIC = 1;
}  // namespace Layers

namespace BroadPhaseLayers
{
static constexpr JPH::BroadPhaseLayer STATIC(0);
static constexpr JPH::BroadPhaseLayer DYNAMIC(1);
}  // namespace BroadPhaseLayers

class ObjectLayerPairFilterImpl : public JPH::ObjectLayerPairFilter
{
   public:
    // Static bodies only collide with dynamic ones.
    bool ShouldCollide(JPH::ObjectLayer a, JPH::ObjectLayer b) const override
    {
        return a == Layers::DYNAMIC || b == Layers::DYNAMIC;
    }
};

class BroadPhaseLayerInterfaceImpl final : public JPH::BroadPhaseLayerInterface
{
   public:
    JPH::uint GetNumBroadPhaseLayers() const override { return 2; }
    JPH::BroadPhaseLayer GetBroadPhaseLayer(JPH::ObjectLayer layer) const override
    {
        return layer == Layers::STATIC ? BroadPhaseLayers::STATIC : BroadPhaseLayers::DYNAMIC;
    }
#if defined(JPH_EXTERNAL_PROFILE) || defined(JPH_PROFILE_ENABLED)
    const char* GetBroadPhaseLayerName(JPH::BroadPhaseLayer layer) const override
    {
        return layer == BroadPhaseLayers::STATIC ? "STATIC" : "DYNAMIC";
    }
#endif
};

class ObjectVsBroadPhaseLayerFilterImpl : public JPH::ObjectVsBroadPhaseLayerFilter
{
   public:
    bool ShouldCollide(JPH::ObjectLayer layer, JPH::BroadPhaseLayer bp_layer) const override
    {
        return layer == Layers::DYNAMIC || bp_layer == BroadPhaseLayers::DYNAMIC;
    }
};

// Registers Jolt's allocator, factory and types for the lifetime of the program.
struct JoltRuntime
{
    JoltRuntime()
    {
        JPH::RegisterDefaultAllocator();
        JPH::Factory::sInstance = new JPH::Factory();
        JPH::RegisterTypes();
    }
    ~JoltRuntime()
    {
        JPH::UnregisterTypes();
        delete JPH::Factory::sInstance;
        JPH::Factory::sInstance = nullptr;
    }
};

class JoltSample : public VulkanExampleBase, public glfw_MouseListener
{
   public:
    JoltSample();
    ~JoltSample() override;

    void update(float dt) override;
    void transfer(vk::CommandBuffer cmd) override;
    void render(vk::CommandBuffer cmd) override;
    void renderGUI() override;
    void keyPressed(int key, int scancode, int mods) override;
    void mousePressed(int key, int x, int y) override;

   private:
    struct RenderBody
    {
        JPH::BodyID id;
        Vulkan::VulkanVertexColoredAsset* mesh;
    };

    void createScene();
    void clearScene();
    void addBox(vec3 position, int color);
    void shootBall(vec3 position, vec3 velocity);
    void shootFromCamera();

    // Physics
    JPH::TempAllocatorImpl tempAllocator{64 * 1024 * 1024};
    std::unique_ptr<JPH::JobSystemThreadPool> jobSystem;
    BroadPhaseLayerInterfaceImpl broadPhaseLayers;
    ObjectVsBroadPhaseLayerFilterImpl objectVsBroadPhaseFilter;
    ObjectLayerPairFilterImpl objectPairFilter;
    std::unique_ptr<JPH::PhysicsSystem> physics;
    JPH::RefConst<JPH::Shape> boxShape, ballShape;
    JPH::BodyID ground;
    std::vector<RenderBody> bodies;
    std::vector<mat4> models;

    // Rendering
    Vulkan::AssetRenderer assetRenderer;
    Vulkan::VulkanVertexColoredAsset plane, ball;
    std::vector<std::unique_ptr<Vulkan::VulkanVertexColoredAsset>> boxMeshes;

    // Settings
    int pyramidRows     = 16;
    float ballSpeed     = 40;
    int numThreads      = 1;
    float stepTimeMS    = 0;
    bool paused         = false;
    bool resetRequested = false;
};

JoltSample::JoltSample()
{
    // Flying first-person camera: both drag buttons look around with the y axis kept up (the default of the second
    // button orbits around a point with z up). SPACE moves up, CTRL down (in update()).
    camera.setProj(60.0f, window->getAspectRatio(), 0.1f, 1000.0f, true);
    camera.setView(vec3(-18, 12, 30), vec3(0, 5, 0), vec3(0, 1, 0));
    camera.mousemap = {GLFW_MOUSE_BUTTON_LEFT, GLFW_MOUSE_BUTTON_RIGHT};
    camera.mode0    = CameraControlMode::ROTATE_FIRST_PERSON_FIX_UP_VECTOR;
    camera.mode1    = CameraControlMode::ROTATE_FIRST_PERSON_FIX_UP_VECTOR;
    auto& base = renderer->base();
    assetRenderer.init(base, renderer->renderingFormats());

    static_cast<TriangleMesh<VertexNC, uint32_t>&>(plane) =
        CheckerBoardPlane(ivec2(200, 200), 1.0f, Colors::gray, Colors::lightgray).Mesh<VertexNC, uint32_t>();
    plane.init(base);

    static_cast<TriangleMesh<VertexNC, uint32_t>&>(ball) =
        IcoSphereMesh(Sphere(make_vec3(0), 0.7f), 3).SetVertexColor(Colors::firebrick).Mesh<VertexNC, uint32_t>();
    ball.init(base);

    for (vec3 c : {vec3(0.85f, 0.65f, 0.35f), vec3(0.75f, 0.5f, 0.25f), vec3(0.6f, 0.62f, 0.66f),
                   vec3(0.45f, 0.55f, 0.7f)})
    {
        auto mesh = std::make_unique<Vulkan::VulkanVertexColoredAsset>();
        static_cast<TriangleMesh<VertexNC, uint32_t>&>(*mesh) =
            BoxMesh(AABB(make_vec3(-0.5f), make_vec3(0.5f))).SetVertexColor(make_vec4(c, 1)).Mesh<VertexNC, uint32_t>();
        mesh->init(base);
        boxMeshes.push_back(std::move(mesh));
    }

    numThreads = std::max(1, int(std::thread::hardware_concurrency()) - 1);
    jobSystem  = std::make_unique<JPH::JobSystemThreadPool>(JPH::cMaxPhysicsJobs, JPH::cMaxPhysicsBarriers, numThreads);
    physics    = std::make_unique<JPH::PhysicsSystem>();
    physics->Init(65536, 0, 65536, 65536, broadPhaseLayers, objectVsBroadPhaseFilter, objectPairFilter);

    boxShape = new JPH::BoxShape(JPH::Vec3::sReplicate(0.5f));
    auto s   = new JPH::SphereShape(0.7f);
    s->SetDensity(20000);  // a heavy ball (about 29 t), the boxes have the default density of 1000 kg/m^3
    ballShape = s;

    createScene();
}

JoltSample::~JoltSample()
{
    renderer->waitIdle();
    clearScene();
    physics->GetBodyInterface().RemoveBody(ground);
    physics->GetBodyInterface().DestroyBody(ground);
}

void JoltSample::addBox(vec3 position, int color)
{
    JPH::BodyCreationSettings settings(boxShape, JPH::RVec3(position.x(), position.y(), position.z()),
                                       JPH::Quat::sIdentity(), JPH::EMotionType::Dynamic, Layers::DYNAMIC);
    settings.mFriction = 0.6f;
    auto id            = physics->GetBodyInterface().CreateAndAddBody(settings, JPH::EActivation::Activate);
    bodies.push_back({id, boxMeshes[color % boxMeshes.size()].get()});
}

void JoltSample::shootBall(vec3 position, vec3 velocity)
{
    JPH::BodyCreationSettings settings(ballShape, JPH::RVec3(position.x(), position.y(), position.z()),
                                       JPH::Quat::sIdentity(), JPH::EMotionType::Dynamic, Layers::DYNAMIC);
    settings.mLinearVelocity = JPH::Vec3(velocity.x(), velocity.y(), velocity.z());
    settings.mMotionQuality  = JPH::EMotionQuality::LinearCast;  // no tunneling through the thin wall
    auto id = physics->GetBodyInterface().CreateAndAddBody(settings, JPH::EActivation::Activate);
    bodies.push_back({id, &ball});
}

void JoltSample::createScene()
{
    auto& bi = physics->GetBodyInterface();
    if (ground.IsInvalid())
    {
        // A large static box with its top face at y = 0.
        ground = bi.CreateAndAddBody(
            JPH::BodyCreationSettings(new JPH::BoxShape(JPH::Vec3(100, 1, 100)), JPH::RVec3(0, -1, 0),
                                      JPH::Quat::sIdentity(), JPH::EMotionType::Static, Layers::STATIC),
            JPH::EActivation::DontActivate);
    }

    // Pyramid in the xy-plane at z = 0.
    for (int row = 0; row < pyramidRows; ++row)
    {
        int n = pyramidRows - row;
        for (int i = 0; i < n; ++i)
        {
            addBox(vec3(i - (n - 1) * 0.5f, 0.5f + row, 0), row);
        }
    }

    // A wall behind the pyramid, bricks offset by half a box every other row.
    for (int row = 0; row < 10; ++row)
    {
        for (int i = 0; i < 14; ++i)
        {
            addBox(vec3(i - 6.5f + (row % 2) * 0.5f, 0.5f + row, -8), 2 + (row % 2));
        }
    }

    physics->OptimizeBroadPhase();

    // Fire the first ball from the front into the lower half of the pyramid.
    vec3 start(4, 4, 24);
    vec3 target(0, 3, 0);
    shootBall(start, (target - start).normalized() * ballSpeed);
}

void JoltSample::clearScene()
{
    auto& bi = physics->GetBodyInterface();
    for (auto& b : bodies)
    {
        bi.RemoveBody(b.id);
        bi.DestroyBody(b.id);
    }
    bodies.clear();
}

void JoltSample::update(float dt)
{
    VulkanExampleBase::update(dt);
    if (!ImGui::captureKeyboard() && keyboard().getKeyState(GLFW_KEY_LEFT_CONTROL))
    {
        float speed = keyboard().getKeyState(GLFW_KEY_LEFT_SHIFT) ? camera.movementSpeedFast : camera.movementSpeed;
        camera.translateGlobal(vec3(0, -speed * dt, 0));
        camera.calculateModel();
        camera.updateFromModel();
    }

    if (resetRequested)
    {
        clearScene();
        createScene();
        resetRequested = false;
    }

    if (!paused)
    {
        auto start = std::chrono::steady_clock::now();
        physics->Update(dt, 1, &tempAllocator, jobSystem.get());
        stepTimeMS = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count();
    }

    // Read the transforms here (single threaded, between the physics steps) for render().
    const auto& bi = physics->GetBodyInterfaceNoLock();
    models.resize(bodies.size());
    for (size_t i = 0; i < bodies.size(); ++i)
    {
        JPH::RMat44 t = bi.GetWorldTransform(bodies[i].id);
        mat4 m;
        for (int r = 0; r < 4; ++r)
        {
            for (int c = 0; c < 4; ++c)
            {
                m(r, c) = float(t(r, c));
            }
        }
        models[i] = m;
    }
}

void JoltSample::transfer(vk::CommandBuffer cmd)
{
    assetRenderer.updateUniformBuffers(cmd, camera.view, camera.proj);
}

void JoltSample::render(vk::CommandBuffer cmd)
{
    if (!assetRenderer.bind(cmd)) return;

    assetRenderer.pushModel(cmd, mat4::Identity());
    plane.render(cmd);

    for (size_t i = 0; i < models.size(); ++i)
    {
        assetRenderer.pushModel(cmd, models[i]);
        bodies[i].mesh->render(cmd);
    }
}

void JoltSample::renderGUI()
{
    ImGui::SetNextWindowSize(ImVec2(300, 200), ImGuiCond_FirstUseEver);
    ImGui::Begin("Jolt Physics");
    ImGui::Text("%d bodies, %d active, %d threads", int(bodies.size()), int(physics->GetNumActiveBodies(JPH::EBodyType::RigidBody)),
                numThreads);
    ImGui::Text("Step: %.2f ms", stepTimeMS);
    ImGui::Checkbox("Pause", &paused);
    ImGui::SliderInt("Pyramid rows", &pyramidRows, 1, 40);
    ImGui::SliderFloat("Ball speed", &ballSpeed, 5, 100);
    if (ImGui::Button("Reset (R)")) resetRequested = true;
    ImGui::Text("Drag: look, WASD: move, SPACE/CTRL: up/down");
    ImGui::Text("F / middle mouse: shoot a ball");
    ImGui::End();

    VulkanExampleBase::renderGUI();
}

void JoltSample::keyPressed(int key, int scancode, int mods)
{
    VulkanExampleBase::keyPressed(key, scancode, mods);
    if (ImGui::captureKeyboard()) return;

    switch (key)
    {
        case GLFW_KEY_F:
            shootFromCamera();
            break;
        case GLFW_KEY_R:
            resetRequested = true;
            break;
        default:
            break;
    }
}

void JoltSample::mousePressed(int key, int x, int y)
{
    if (ImGui::captureMouse()) return;
    if (key == GLFW_MOUSE_BUTTON_MIDDLE) shootFromCamera();
}

void JoltSample::shootFromCamera()
{
    // The camera looks along its negative z axis.
    vec3 dir = -make_vec3(camera.getDirection()).normalized();
    shootBall(camera.getPosition() + dir, dir * ballSpeed);
}

int main(const int argc, const char* argv[])
{
    initSaigaSample();
    JoltRuntime jolt;
    {
        JoltSample sample;
        sample.run();
    }
    return 0;
}
