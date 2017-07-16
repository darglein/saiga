#pragma once

#include <saiga/assets/asset.h>
#include <saiga/opengl/texture/texture.h>
#include "saiga/opengl/uniformBuffer.h"

namespace Saiga {

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

class SAIGA_GLOBAL AnimatedAsset : public BasicAsset<BoneVertexCD,GLuint>{
public:
    int boneCount;

    std::map<std::string,int> boneMap;
    std::map<std::string,int> nodeindexMap;

    std::vector<mat4> boneOffsets;
    std::vector<mat4> inverseBoneOffsets;

    std::vector<Animation> animations;


    void render(Camera *cam, const mat4 &model, UniformBuffer& boneMatrices);
    void renderDepth(Camera *cam, const mat4 &model, UniformBuffer& boneMatrices);
};

}
