/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/config.h"
#include "saiga/core/geometry/object3d.h"
#include "saiga/core/window/Interfaces.h"
#include "saiga/opengl/assets/all.h"

#include <memory>

namespace Saiga
{
class SAIGA_OPENGL_API AssetRenderSystem
{
    template <typename AssetType>
    struct RenderData
    {
        AssetType* asset;
        mat4 model;
        int flags;
    };

   public:
    AssetRenderSystem();
    void Clear();
    void Render(RenderInfo render_info);
    void Add(Asset* asset, const mat4& transformation, int render_flags);

   private:
    std::vector<RenderData<ColoredAsset>> colored_assets;
    std::vector<RenderData<TexturedAsset>> textured_assets;

    std::shared_ptr<MVPColorShader> shader_colored_deferred;
    std::shared_ptr<MVPColorShader> shader_colored_forward;
    std::shared_ptr<MVPColorShader> shader_colored_depth;

    std::shared_ptr<MVPTextureShader> shader_textured_deferred;
    std::shared_ptr<MVPTextureShader> shader_textured_forward;
    std::shared_ptr<MVPTextureShader> shader_textured_depth;

};

}  // namespace Saiga
