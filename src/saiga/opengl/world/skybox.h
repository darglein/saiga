/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/opengl/UnifiedMeshBuffer.h"
#include "saiga/opengl/indexedVertexBuffer.h"
#include "saiga/opengl/shader/basic_shaders.h"
#include "saiga/opengl/texture/CubeTexture.h"
#include "saiga/opengl/texture/Texture.h"
#include "saiga/opengl/vertex.h"
namespace Saiga
{
class SAIGA_OPENGL_API Skybox
{
   public:
    Skybox(std::shared_ptr<Texture> texture, const std::string mapping = "spherical");
    void render(Camera* cam);

   protected:
    int type = -1;
    UnifiedMeshBuffer mesh;
    std::shared_ptr<MVPTextureShader> shader;
    std::shared_ptr<Texture> texture;
    std::shared_ptr<TextureCube> cube_texture;
};

}  // namespace Saiga
