/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/opengl/shader/basic_shaders.h"
#include "saiga/opengl/uniformBuffer.h"

namespace Saiga
{
#define BONE_MATRICES_BINDING_POINT 1

class SAIGA_OPENGL_API BoneShader : public MVPShader
{
   public:
    GLint location_boneMatrices;
    GLint location_boneMatricesBlock;


    virtual void checkUniforms();

    void uploadBoneMatrices(mat4* matrices, int count);
};

}  // namespace Saiga
