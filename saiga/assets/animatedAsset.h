#pragma once

#include <saiga/assets/asset.h>
#include <saiga/opengl/texture/texture.h>
#include "saiga/opengl/uniformBuffer.h"

/**
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
 */

class SAIGA_GLOBAL AnimatedAsset : public BasicAsset<BoneVertexNC,GLuint>{
public:



    int boneCount;

    std::map<std::string,int> boneMap;
    std::map<std::string,int> nodeindexMap;

    std::vector<mat4> boneOffsets;
    std::vector<mat4> inverseBoneOffsets;

    std::vector<Animation> animations;
    std::vector<float> animationSpeeds;





    void render(Camera *cam, const mat4 &model, UniformBuffer& boneMatrices);
    void renderDepth(Camera *cam, const mat4 &model, UniformBuffer& boneMatrices);
};
