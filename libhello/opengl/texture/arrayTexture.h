#pragma once

#include "libhello/opengl/texture/raw_texture.h"


class ArrayTexture2D : public raw_Texture{

public:
    int depth;

    ArrayTexture2D():raw_Texture(GL_TEXTURE_2D_ARRAY){}
    virtual ~ArrayTexture2D(){}


     void setDefaultParameters() override;


    void uploadData(GLenum target, GLubyte* data );
    void uploadData(GLubyte** data );
    void uploadData(GLubyte* data);

    bool fromImage(std::vector<Image> &images);
    bool fromImage(Image &img);
};

