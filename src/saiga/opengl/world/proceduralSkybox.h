/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/camera/camera.h"
#include "saiga/core/rendering/ProceduralSkyboxBase.h"
#include "saiga/opengl/indexedVertexBuffer.h"
#include "saiga/opengl/shader/basic_shaders.h"
#include "saiga/opengl/texture/CubeTexture.h"
#include "saiga/opengl/texture/Texture.h"
#include "saiga/opengl/vertex.h"
#include "saiga/opengl/UnifiedMeshBuffer.h"

namespace Saiga
{
class SAIGA_OPENGL_API ProceduralSkybox : public ProceduralSkyboxBase
{
   public:
    ProceduralSkybox(const std::string& shader_str = "geometry/skybox_procedural.glsl");
    void render(Camera* cam, const mat4& model = mat4::Identity());

   protected:
    UnifiedMeshBuffer mesh;
    std::shared_ptr<MVPShader> shader;
};

}  // namespace Saiga
