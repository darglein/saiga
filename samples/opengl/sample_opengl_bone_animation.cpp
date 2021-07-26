/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/model/all.h"
#include "saiga/opengl/window/SampleWindowDeferred.h"
#include "saiga/opengl/window/message_box.h"
using namespace Saiga;

class Sample : public SampleWindowDeferred
{
    using Base = SampleWindowDeferred;

   public:
    Sample()
    {
        auto model = UnifiedModel("animation_sample/idle.dae");
        auto anim1 = UnifiedModel("animation_sample/forward.dae");
        auto anim2 = UnifiedModel("animation_sample/left.dae");
        auto anim3 = UnifiedModel("animation_sample/right.dae");
        auto anim4 = UnifiedModel("animation_sample/backward.dae");

        model.animation_system.animations.push_back(anim1.animation_system.animations[0]);
        model.animation_system.animations.push_back(anim2.animation_system.animations[0]);
        model.animation_system.animations.push_back(anim3.animation_system.animations[0]);
        model.animation_system.animations.push_back(anim4.animation_system.animations[0]);


        asset             = std::make_shared<AnimatedAsset>(model);
        vec_bone_matrices = AlignedVector<mat4>(asset->animation_system.boneOffsets.size(), mat4::Identity());
        bone_matrices.createGLBuffer(vec_bone_matrices.data(), vec_bone_matrices.size() * sizeof(mat4));
        bone_matrices.bind(1);

        camera.position = vec4(3.62443, 3.02657, 1.68532, 1);
        camera.rot      = quat(0.794069, -0.253948, 0.517145, 0.193717);

        std::cout << "Program Initialized!" << std::endl;
    }
    void update(float dt) override
    {
        Base::update(dt);
        int animation = 0;

        if (keyboard.getKeyState(GLFW_KEY_UP))
        {
            animation = 1;
        }

        if (keyboard.getKeyState(GLFW_KEY_LEFT))
        {
            animation = 2;
        }
        if (keyboard.getKeyState(GLFW_KEY_RIGHT))
        {
            animation = 3;
        }
        if (keyboard.getKeyState(GLFW_KEY_DOWN))
        {
            animation = 4;
        }

        asset->animation_system.SetAnimation(animation, interpolate_between_animtation);
        asset->animation_system.update(dt);
    }
    void interpolate(float dt, float alpha) override
    {
        Base::interpolate(dt, alpha);
        asset->animation_system.interpolate(dt, alpha);
        vec_bone_matrices = asset->animation_system.Matrices();
        bone_matrices.updateBuffer(vec_bone_matrices.data(), vec_bone_matrices.size() * sizeof(mat4), 0);
    }
    void render(RenderInfo render_info) override
    {
        Base::render(render_info);
        if (render_info.render_pass == RenderPass::Deferred || render_info.render_pass == RenderPass::Shadow)
        {
            asset->render(render_info.camera, mat4::Identity());
        }

        if (render_info.render_pass == RenderPass::GUI)
        {
            ImGui::SetNextWindowPos(ImVec2(0, 200), ImGuiCond_Once);
            ImGui::SetNextWindowSize(ImVec2(400, 300), ImGuiCond_Once);
            ImGui::Begin("Animation Sample");
            ImGui::Text("Controls");
            ImGui::Text("   Forward:  [Arrow Up]");
            ImGui::Text("   Backward: [Arrow Down]");
            ImGui::Text("   Left:     [Arrow Left]");
            ImGui::Text("   Right:    [Arrow Right]");

            ImGui::Checkbox("interpolate_between_animtation", &interpolate_between_animtation);
            asset->animation_system.imgui();
            ImGui::End();
        }
    }



   private:
    AlignedVector<mat4> vec_bone_matrices;
    std::shared_ptr<AnimatedAsset> asset;
    UniformBuffer bone_matrices;
    bool interpolate_between_animtation = true;
};



int main(int argc, char* args[])
{
    // This should be only called if this is a sample located in saiga/samples
    initSaigaSample();
    Sample window;
    window.run();
    return 0;
}
