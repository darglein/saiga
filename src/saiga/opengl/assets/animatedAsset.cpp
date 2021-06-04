/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "animatedAsset.h"

#include "saiga/opengl/animation/boneShader.h"
#include "saiga/opengl/shader/shaderLoader.h"

namespace Saiga
{
void AnimatedAsset::loadDefaultShaders()
{
    this->deferredShader  = shaderLoader.load<BoneShader>(shaderStr, {{GL_FRAGMENT_SHADER, "#define DEFERRED", 1}});
    this->forwardShader   = shaderLoader.load<BoneShader>(shaderStr, {{GL_FRAGMENT_SHADER, "#define DEFERRED", 1}});
    this->depthshader     = shaderLoader.load<BoneShader>(shaderStr);
    this->wireframeshader = shaderLoader.load<BoneShader>(shaderStr);
}

AnimatedAsset::AnimatedAsset(const UnifiedModel& model)
{
//    auto mesh      = model.mesh[0].Mesh<BoneVertexCD, uint32_t>();
//    this->vertices = mesh.vertices;
//    this->faces    = mesh.faces;
    auto [mesh, groups] = model.CombinedMesh(VERTEX_POSITION | VERTEX_NORMAL | VERTEX_COLOR | VERTEX_BONE_INFO);
    this->groups        = groups;
    unified_buffer = std::make_shared<UnifiedMeshBuffer>(mesh);
    loadDefaultShaders();

    animation_system = model.animation_system;

//    std::cout << "Create AnimatedAsset " << vertices.size() << " " << faces.size() << std::endl;
//    create();
}

#if 0
void AnimatedAsset::render(Camera* cam, const mat4& model, UniformBuffer& boneMatrices)
{
    std::shared_ptr<BoneShader> bs = std::static_pointer_cast<BoneShader>(this->deferredShader);

    std::terminate();
    bs->bind();
    bs->uploadModel(model);
    //    boneMatrices.bind(0);
    //    boneMatrices.bind(BONE_MATRICES_BINDING_POINT);
    buffer.bindAndDraw();
    bs->unbind();
}

void AnimatedAsset::renderDepth(Camera* cam, const mat4& model, UniformBuffer& boneMatrices)
{
    std::shared_ptr<BoneShader> bs = std::static_pointer_cast<BoneShader>(this->depthshader);

    bs->bind();
    bs->uploadModel(model);
    //    boneMatrices.bind(BONE_MATRICES_BINDING_POINT);
    buffer.bindAndDraw();
    bs->unbind();
}
#endif

}  // namespace Saiga
