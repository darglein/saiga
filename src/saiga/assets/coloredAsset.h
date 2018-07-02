/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <saiga/assets/asset.h>
#include <saiga/opengl/texture/texture.h>

namespace Saiga {

class SAIGA_GLOBAL ColoredAsset : public BasicAsset<VertexColoredModel>{
};

class SAIGA_GLOBAL TexturedAsset : public BasicAsset<TexturedModel>{
public:
    class SAIGA_GLOBAL TextureGroup{
    public:
        int startIndex;
        int indices;
        std::shared_ptr<Texture> texture;
    };
    std::vector<TextureGroup> groups;

    virtual void render(Camera *cam, const mat4 &model) override;
    virtual void renderForward(Camera *cam, const mat4 &model) override;
    virtual void renderDepth(Camera *cam, const mat4 &model) override;

};

}
