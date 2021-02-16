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
class SAIGA_OPENGL_API ColoredAsset : public BasicAsset<VertexColoredModel, MVPColorShader>
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

class SAIGA_OPENGL_API LineVertexColoredAsset : public BasicAsset<LineMesh<VertexC, uint32_t>, MVPColorShader>
{
   public:
    // Default shaders
    // If you want to use your own load them and override the shader memebers in BasicAsset.
    static constexpr const char* shaderStr = "asset/LineVertexColoredAsset.glsl";
    void loadDefaultShaders();

    void SetShaderColor(const vec4& color);

    virtual ~LineVertexColoredAsset() {}
};


class SAIGA_OPENGL_API TexturedAsset : public BasicAsset<TexturedModel, MVPTextureShader>
{
   public:
    // Default shaders
    // If you want to use your own load them and override the shader memebers in BasicAsset.
    static constexpr const char* shaderStr = "asset/texturedAsset.glsl";
    void loadDefaultShaders() override;


    //    class SAIGA_OPENGL_API TextureGroup
    //    {
    //       public:
    //        int startIndex;
    //        int indices;
    //        std::shared_ptr<Texture> texture;
    //    };
    //    std::vector<TextureGroup> groups;


    std::vector<UnifiedMaterial> materials;
    std::vector<UnifiedMaterialGroup> groups;

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
