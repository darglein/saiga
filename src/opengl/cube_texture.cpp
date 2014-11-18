
#include "opengl/cube_texture.h"


void cube_Texture::uploadData(GLubyte* data ){
    bind(0);
    for (int i=0; i<6; i++) {
      glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                 0,  // level, 0 = base, no minimap,
                 internal_format, // internalformat
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

void cube_Texture::setDefaultParameters(){
    glTexParameteri (target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri (target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri (target, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri (target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri (target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}




