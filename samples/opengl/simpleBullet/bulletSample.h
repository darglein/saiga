/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once


#include "saiga/opengl/window/SampleWindowDeferred.h"

#include "physics.h"
using namespace Saiga;


struct PhysicAssetObject : public SimpleAssetObject
{
    btRigidBody* rigidBody;

    void loadFromRigidbody()
    {
        btTransform trans;
        rigidBody->getMotionState()->getWorldTransform(trans);
        setPosition(toGLM(trans.getOrigin()));
        rot = toGLM(trans.getRotation());
    }
};

class Sample : public SampleWindowDeferred
{
   public:
    BulletPhysics physics;
    std::vector<PhysicAssetObject> cubes;

    std::shared_ptr<Asset> cubeAsset;



    Sample();
    ~Sample();

    void initBullet();

    void update(float dt) override;

    void render(RenderInfo render_info) override
    {
        SampleWindowDeferred::render(render_info);
        if (render_info.render_pass == RenderPass::Deferred ||render_info. render_pass == RenderPass::Shadow)
        {
            for (auto& cube : cubes) cube.render(render_info.camera);
        }
        else if (render_info.render_pass == RenderPass::Forward)
        {
            physics.render(render_info.camera);
        }
        else if (render_info.render_pass == RenderPass::GUI)
        {
            ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowSize(ImVec2(400, 200), ImGuiCond_FirstUseEver);
            ImGui::Begin("An Imgui Window :D");

            ImGui::End();
        }
    }
};
