/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/opengl/assets/asset.h"
#include "saiga/opengl/texture/Texture.h"


namespace Saiga
{
class SAIGA_OPENGL_API ColoredAsset : public BasicAsset< MVPColorShader>
{
   public:
    static constexpr const char* shaderStr = "asset/ColoredAsset.glsl";
    void loadDefaultShaders();

    ColoredAsset() { loadDefaultShaders(); }
    ColoredAsset(const UnifiedModel& model);
    ColoredAsset(const UnifiedMesh& model);


    virtual ~ColoredAsset() {}
};

class SAIGA_OPENGL_API LineVertexColoredAsset : public BasicAsset<MVPColorShader>
{
   public:
    enum RenderFlags
    {
        NONE             = 0,
        CULL_WITH_NORMAL = 1 << 1
    };

    // Default shaders
    // If you want to use your own load them and override the shader memebers in BasicAsset.
    static constexpr const char* shaderStr = "asset/LineVertexColoredAsset.glsl";
    void loadDefaultShaders();

    void SetShaderColor(const vec4& color);
    void SetRenderFlags(RenderFlags flags = NONE);

    LineVertexColoredAsset() { loadDefaultShaders(); }
    LineVertexColoredAsset(const UnifiedMesh& model);

    virtual ~LineVertexColoredAsset() {}
};


class SAIGA_OPENGL_API TexturedAsset : public BasicAsset< MVPTextureShader>
{
   public:
    // Default shaders
    // If you want to use your own load them and override the shader memebers in BasicAsset.
    static constexpr const char* shaderStr = "asset/texturedAsset.glsl";
    void loadDefaultShaders() override;

    std::vector<UnifiedMaterial> materials;

    std::vector<std::shared_ptr<Texture>> textures;
    std::map<std::string, int> texture_name_to_id;

    TexturedAsset() { loadDefaultShaders(); }
    TexturedAsset(const UnifiedModel& model);

    virtual ~TexturedAsset() {}


    virtual void render(Camera* cam, const mat4& model) override;
    virtual void renderForward(Camera* cam, const mat4& model) override;
    virtual void renderDepth(Camera* cam, const mat4& model) override;

    void RenderNoShaderBind(MVPTextureShader* shader);
   protected:
    void renderGroups(std::shared_ptr<MVPTextureShader> shader, Camera* cam, const mat4& model);

};

}  // namespace Saiga
