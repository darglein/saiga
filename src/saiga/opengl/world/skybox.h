/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/opengl/indexedVertexBuffer.h"
#include "saiga/opengl/shader/basic_shaders.h"
#include "saiga/opengl/texture/CubeTexture.h"
#include "saiga/opengl/texture/Texture.h"
#include "saiga/opengl/vertex.h"
#include "saiga/opengl/UnifiedMeshBuffer.h"
namespace Saiga
{
class SAIGA_OPENGL_API Skybox
{
   public:
    UnifiedMeshBuffer mesh;
    std::shared_ptr<MVPTextureShader> shader;
    std::shared_ptr<Texture> texture;
    std::shared_ptr<TextureCube> cube_texture;
    mat4 model = mat4::Identity();

    Skybox();

    void setPosition(const vec3& p);
    void setDistance(float d);
    void render(Camera* cam);
};

}  // namespace Saiga
