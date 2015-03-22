
#include "opengl/texture/arrayTexture.h"
#include "libhello/util/error.h"
void ArrayTexture2D::uploadData(GLenum target, GLubyte *data ){
    bind(0);
    glTexImage2D(target,
                 0,  // level, 0 = base, no minimap,
                 internal_format, // internalformat
                 width,  // width
                 height,  // height
                 0,  // border, always 0 in OpenGL ES
                 color_type,  // format
                 data_type, // type
                 data);


    Error::quitWhenError("uploadData()");
}


void ArrayTexture2D::uploadData(GLubyte *data ){
    cout<<">>>>> uploadData"<<endl;
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


void ArrayTexture2D::uploadData(GLubyte **data ){
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
                     data[i]);
    }

    unbind();
    Error::quitWhenError("uploadData()");
}

void ArrayTexture2D::setDefaultParameters(){
    glTexParameteri (target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri (target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri (target, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri (target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri (target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}


bool ArrayTexture2D::fromImage(std::vector<Image> &images){
    depth = images.size();
    setFormat(images[0]);


    createGlTexture();
    bind(0);
    glTexStorage3D(target, 1, internal_format, width, height, depth);

//    cout<<"format: "<<internal_format<<endl;
    Error::quitWhenError("ArrayTexture2D::fromImage2");
    for(int i=0;i<depth;i++){
        GLubyte*  data = images[i].data;
        uploadSubImage(0,0,i,width,height,1,data);
    }


    unbind();
    return true;
}

