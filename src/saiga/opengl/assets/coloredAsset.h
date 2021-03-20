/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/opengl/assets/asset.h"
#include "saiga/opengl/texture/Texture.h"


namespace Saiga
{
class SAIGA_OPENGL_API ColoredAsset : public BasicAsset<TriangleMesh<VertexNC, uint32_t>, MVPColorShader>
{
   public:
    static constexpr const char* shaderStr = "asset/ColoredAsset.glsl";
    void loadDefaultShaders();

    ColoredAsset() {}


    ColoredAsset(const TriangleMesh<VertexNC, uint32_t>& mesh);
    ColoredAsset(const UnifiedModel& model);
    ColoredAsset(const std::string& file) : ColoredAsset(UnifiedModel(file)) {}


    virtual ~ColoredAsset() {}
};

class SAIGA_OPENGL_API LineVertexColoredAsset : public BasicAsset<LineMesh<VertexNC, uint32_t>, MVPColorShader>
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

    LineVertexColoredAsset() {}
    LineVertexColoredAsset(const LineMesh<VertexNC, uint32_t>& line_mesh);
    LineVertexColoredAsset(const UnifiedModel& model);

    virtual ~LineVertexColoredAsset() {}
};


class SAIGA_OPENGL_API TexturedAsset : public BasicAsset<TriangleMesh<VertexNTD, uint32_t>, MVPTextureShader>
{
   public:
    // Default shaders
    // If you want to use your own load them and override the shader memebers in BasicAsset.
    static constexpr const char* shaderStr = "asset/texturedAsset.glsl";
    void loadDefaultShaders() override;


    std::vector<UnifiedMaterialGroup> groups;
    std::vector<UnifiedMaterial> materials;

    std::vector<std::shared_ptr<Texture> > textures;

    TexturedAsset() {}
    TexturedAsset(const UnifiedModel& model);
    TexturedAsset(const std::string& file) : TexturedAsset(UnifiedModel(file)) {}

    virtual ~TexturedAsset() {}


    virtual void render(Camera* cam, const mat4& model) override;
    virtual void renderForward(Camera* cam, const mat4& model) override;
    virtual void renderDepth(Camera* cam, const mat4& model) override;

   protected:
    void renderGroups(std::shared_ptr<MVPTextureShader> shader, Camera* cam, const mat4& model);
};

}  // namespace Saiga
