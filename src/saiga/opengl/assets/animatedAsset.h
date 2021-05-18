/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/opengl/assets/asset.h"
#include "saiga/opengl/texture/Texture.h"
#include "saiga/opengl/uniformBuffer.h"

namespace Saiga
{
/**
 *
 * Select Armature and click Rest Position.
 *
 * Blender Collada animation export options: X = checked, O = unchecked
 *
 * X Apply Modifiers
 * X Selection Only
 * X Include Children
 * X Include Armature
 * X Include Shape Keys
 *
 * 0 Deform Bones only
 * 0 Export for Open Sim
 *
 * Note: The blender animation speed is equal to the loaded animation speed.
 * You can change blender's frame rate in the render options.
 * So if you set blender's frame rate to 30 fps and have your animation start at frame 0 and end at frame 30
 * the animation will play for exactly one second.
 *
 */

class SAIGA_OPENGL_API AnimatedAsset : public BasicAsset<MVPShader>
{
   public:
    // Default shaders
    // If you want to use your own load them and override the shader memebers in BasicAsset.
    //    static constexpr const char* shaderStr = "asset/AnimatedAsset.glsl";
    static constexpr const char* shaderStr = "asset/AnimatedAsset.glsl";

    void loadDefaultShaders() override;

    AnimationSystem animation_system;

    AnimatedAsset() {}
    AnimatedAsset(const UnifiedModel& model);


//    void render(Camera* cam, const mat4& model, UniformBuffer& boneMatrices);
//    void renderDepth(Camera* cam, const mat4& model, UniformBuffer& boneMatrices);
};

}  // namespace Saiga
