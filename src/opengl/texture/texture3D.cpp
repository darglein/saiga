#include "saiga/opengl/texture/texture3D.h"
#include "saiga/util/error.h"

Texture3D::Texture3D(GLenum target) : raw_Texture(target)
{
    assert(target == GL_TEXTURE_3D || target == GL_TEXTURE_2D_ARRAY);
}

void Texture3D::setDefaultParameters(){
    glTexParameteri (target, GL_TEXTURE_MAG_FILTER, static_cast<GLint>(GL_LINEAR));
    glTexParameteri (target, GL_TEXTURE_MIN_FILTER, static_cast<GLint>(GL_LINEAR));
    glTexParameteri (target, GL_TEXTURE_WRAP_R, static_cast<GLint>(GL_CLAMP_TO_EDGE));
    glTexParameteri (target, GL_TEXTURE_WRAP_S, static_cast<GLint>(GL_CLAMP_TO_EDGE));
    glTexParameteri (target, GL_TEXTURE_WRAP_T, static_cast<GLint>(GL_CLAMP_TO_EDGE));
}

void Texture3D::uploadSubImage(int x, int y, int z, int width, int height , int depth, GLubyte* data ){
    bind();
    glTexSubImage3D(target, 0, x, y, z,width, height,depth, color_type, data_type, data);
    assert_no_glerror();
    unbind();
}


bool Texture3D::fromImage(std::vector<Image> &images){
    depth = images.size();
    setFormat(images[0]);


    createGlTexture();
    bind(0);
//    glTexStorage3D(target, 1, internal_format, width, height, depth);
    glTexImage3D(target, 0, internal_format, width, height, depth,0,color_type,data_type,nullptr);

    assert_no_glerror();
    for(int i=0;i<depth;i++){
        Image& img = images[i];
        //make sure all images have the same format
        assert(width == img.width && height == img.height && internal_format == img.Format().getGlInternalFormat());
        uploadSubImage(0,0,i,width,height,1,img.getRawData());
    }


    unbind();
    return true;
}

