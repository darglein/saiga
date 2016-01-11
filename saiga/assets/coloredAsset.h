#pragma once

#include <saiga/assets/asset.h>
#include <saiga/opengl/texture/texture.h>

class SAIGA_GLOBAL ColoredAsset : public BasicAsset<VertexNC,GLuint>{
};


class SAIGA_GLOBAL TexturedAsset : public BasicAsset<VertexNT,GLuint>{
public:
    class SAIGA_GLOBAL TextureGroup{
    public:
        int startIndex;
        int indices;
        Texture* texture;
    };
    std::vector<TextureGroup> groups;

    virtual void render(Camera *cam, const mat4 &model) override;
    virtual void renderDepth(Camera *cam, const mat4 &model) override;

};

class SAIGA_GLOBAL AnimatedAsset : public BasicAsset<VertexNC,GLuint>{
};
