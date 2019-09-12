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
class SAIGA_OPENGL_API ColoredAsset : public BasicAsset<VertexColoredModel>
{
   public:
    // Default shaders
    // If you want to use your own load them and override the shader memebers in BasicAsset.
    static constexpr const char* deferredShaderStr  = "geometry/deferred_mvp_model.glsl";
    static constexpr const char* forwardShaderStr   = "geometry/deferred_mvp_model_forward.glsl";
    static constexpr const char* depthShaderStr     = "geometry/deferred_mvp_model_depth.glsl";
    static constexpr const char* wireframeShaderStr = "geometry/deferred_mvp_model_wireframe.glsl";
    void loadDefaultShaders();


    virtual ~ColoredAsset() {}
};

class SAIGA_OPENGL_API LineVertexColoredAsset : public BasicAsset<LineMesh<VertexC, uint32_t>>
{
   public:
    // Default shaders
    // If you want to use your own load them and override the shader memebers in BasicAsset.
    static constexpr const char* deferredShaderStr  = "colored_points.glsl";
    static constexpr const char* forwardShaderStr   = "colored_points.glsl";
    static constexpr const char* depthShaderStr     = "colored_points.glsl";
    static constexpr const char* wireframeShaderStr = "colored_points.glsl";
    void loadDefaultShaders();


    virtual ~LineVertexColoredAsset() {}
};


class SAIGA_OPENGL_API TexturedAsset : public BasicAsset<TexturedModel>
{
   public:
    // Default shaders
    // If you want to use your own load them and override the shader memebers in BasicAsset.
    static constexpr const char* deferredShaderStr  = "geometry/texturedAsset.glsl";
    static constexpr const char* forwardShaderStr   = "geometry/texturedAsset.glsl";
    static constexpr const char* depthShaderStr     = "geometry/texturedAsset_depth.glsl";
    static constexpr const char* wireframeShaderStr = "geometry/texturedAsset.glsl";
    void loadDefaultShaders();


    class SAIGA_OPENGL_API TextureGroup
    {
       public:
        int startIndex;
        int indices;
        std::shared_ptr<Texture> texture;
    };
    std::vector<TextureGroup> groups;

    virtual ~TexturedAsset() {}
    virtual void render(Camera* cam, const mat4& model) override;
    virtual void renderForward(Camera* cam, const mat4& model) override;
    virtual void renderDepth(Camera* cam, const mat4& model) override;
};

}  // namespace Saiga
