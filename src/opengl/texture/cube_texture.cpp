
#include "opengl/texture/cube_texture.h"
#include "libhello/util/error.h"
void cube_Texture::uploadData(GLenum target, GLubyte *data ){
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


    Error::quitWhenError("uploadData()");
}


void cube_Texture::uploadData(GLubyte *data ){
    std::cout<<">>>>> uploadData"<<std::endl;
    bind(0);
    for (int i=0; i<6; i++) {
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                     0,  // level, 0 = base, no minimap,
                     static_cast<GLint>(internal_format), // internalformat
                     width,  // width
                     height,  // height
                     0,  // border, always 0 in OpenGL ES
                     color_type,  // format
                     data_type, // type
                     data);
    }

    unbind();
    Error::quitWhenError("uploadData()");
}


void cube_Texture::uploadData(GLubyte **data ){
    bind(0);
    for (int i=0; i<6; i++) {
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                     0,  // level, 0 = base, no minimap,
                     static_cast<GLint>(internal_format), // internalformat
                     width,  // width
                     height,  // height
                     0,  // border, always 0 in OpenGL ES
                     color_type,  // format
                     data_type, // type
                     data[i]);
    }

    unbind();
    Error::quitWhenError("uploadData()");
}

void cube_Texture::setDefaultParameters(){
    glTexParameteri (target, GL_TEXTURE_MAG_FILTER, static_cast<GLint>(GL_LINEAR));
    glTexParameteri (target, GL_TEXTURE_MIN_FILTER, static_cast<GLint>(GL_LINEAR));
    glTexParameteri (target, GL_TEXTURE_WRAP_R, static_cast<GLint>(GL_CLAMP_TO_EDGE));
    glTexParameteri (target, GL_TEXTURE_WRAP_S, static_cast<GLint>(GL_CLAMP_TO_EDGE));
    glTexParameteri (target, GL_TEXTURE_WRAP_T, static_cast<GLint>(GL_CLAMP_TO_EDGE));
}


bool cube_Texture::fromImage(Image *img){
    setFormat(img[0]);

    GLubyte* data[6];
    for(int i=0;i<6;i++){
        data[i] = img[i].data;
    }
    createGlTexture();
    uploadData(data);
    return true;
}

bool cube_Texture::fromImage(Image &img){
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
    Image images[6];
    auto w = img.height;
    for(int i=0;i<6;i++){
        img.createSubImage(w*i,0,w,w,images[i]);
        //        images[i] = img;
    }


    setFormat(images[0]);

    createGlTexture();

    uploadData(GL_TEXTURE_CUBE_MAP_POSITIVE_X,images[1].data);
    uploadData(GL_TEXTURE_CUBE_MAP_POSITIVE_Z,images[0].data);
    uploadData(GL_TEXTURE_CUBE_MAP_NEGATIVE_X,images[3].data);
    uploadData(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z,images[2].data);

    uploadData(GL_TEXTURE_CUBE_MAP_POSITIVE_Y,images[4].data);
    uploadData(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,images[5].data);

    return true;
}

