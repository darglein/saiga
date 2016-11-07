#pragma once

#include "saiga/opengl/texture/raw_texture.h"

#include <vector>

class SAIGA_GLOBAL Texture3D : public raw_Texture{

public:
    int depth;

    Texture3D(GLenum target=GL_TEXTURE_3D);
    virtual ~Texture3D(){}

    void uploadSubImage(int x, int y, int z, int width, int height, int depth, GLubyte *data);

    void setDefaultParameters() override;

    bool fromImage(std::vector<Image> &images);
};

