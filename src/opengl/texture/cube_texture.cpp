
#include "saiga/opengl/texture/cube_texture.h"
#include "saiga/util/error.h"


void TextureCube::uploadData(GLenum target,const  GLubyte *data ){
    bind(0);
    glTexImage2D(target,
                 0,  // level, 0 = base, no minimap,
                 static_cast<GLint>(internal_format), // internalformat
                 width,  // width
                 height,  // height
                 0,  // border, always 0 in OpenGL ES
                 color_type,  // format
                 data_type, // type
                 data);

    assert_no_glerror();
}



void TextureCube::uploadData(const GLubyte* data ){
    bind();
    for(int i = 0 ; i < 6 ; ++i){
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X+i, // target
                     0,  // level, 0 = base, no minimap,
                     static_cast<GLint>(internal_format), // internalformat
                     width,  // width
                     height,  // height
                     0,
                     color_type,  // format
                     data_type, // type
                     data);
    }
    assert_no_glerror();
    unbind();
}


void TextureCube::setDefaultParameters(){
    glTexParameteri (target, GL_TEXTURE_MAG_FILTER, static_cast<GLint>(GL_LINEAR));
    glTexParameteri (target, GL_TEXTURE_MIN_FILTER, static_cast<GLint>(GL_LINEAR));
    glTexParameteri (target, GL_TEXTURE_WRAP_R, static_cast<GLint>(GL_CLAMP_TO_EDGE));
    glTexParameteri (target, GL_TEXTURE_WRAP_S, static_cast<GLint>(GL_CLAMP_TO_EDGE));
    glTexParameteri (target, GL_TEXTURE_WRAP_T, static_cast<GLint>(GL_CLAMP_TO_EDGE));
}


bool TextureCube::fromImage(Image &img){
    //cubestrip
    if(img.width%6!=0){
        std::cout<<"Width no factor of 6!"<<std::endl;
        return false;
    }

    if(img.width/6 != img.height){
        std::cout<<"No square!"<<std::endl;
        return false;
    }

    //split into 6 small images
    std::vector<Image> images(6);
    auto w = img.height;
    for(int i=0;i<6;i++){
        img.getSubImage(w*i,0,w,w,images[i]);
    }

    return fromImage(images);
}

bool TextureCube::fromImage(std::vector<Image> &images)
{
    setFormat(images[0]);

    createGlTexture();

    uploadData(GL_TEXTURE_CUBE_MAP_POSITIVE_X,images[1].getRawData());
    uploadData(GL_TEXTURE_CUBE_MAP_POSITIVE_Z,images[0].getRawData());
    uploadData(GL_TEXTURE_CUBE_MAP_NEGATIVE_X,images[3].getRawData());
    uploadData(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z,images[2].getRawData());

    uploadData(GL_TEXTURE_CUBE_MAP_POSITIVE_Y,images[4].getRawData());
    uploadData(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,images[5].getRawData());

    assert_no_glerror();
    return true;
}

