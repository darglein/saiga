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
/**
 * Renders a texture into the given screen-space viewport.
 * The texture will be stretched into the given size.
 */
class SAIGA_OPENGL_API TextureDisplay
{
   public:
    TextureDisplay();
    void render(TextureBase* texture, const ivec2& position, const ivec2& size, bool flip_y = false);

   private:
    std::shared_ptr<MVPTextureShader> shader;
    UnifiedMeshBuffer buffer;
};

}  // namespace Saiga
