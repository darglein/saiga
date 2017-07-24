/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/texture/raw_texture.h"
#include "saiga/util/error.h"

namespace Saiga {

raw_Texture::~raw_Texture(){
    deleteGlTexture();
}

void raw_Texture::createTexture(int width, int height, GLenum color_type, GLenum internal_format, GLenum data_type){
    this->width = width;
    this->height = height;
    this->color_type = color_type;
    this->data_type = data_type;
    this->internal_format = internal_format;
    createGlTexture();
}

void raw_Texture::createTexture(int width, int height, GLenum color_type, GLenum internal_format, GLenum data_type,const  GLubyte* data){
    createTexture(width,height,color_type, internal_format,  data_type);
    uploadData(data);
}


void raw_Texture::createEmptyTexture(int width, int height, GLenum color_type, GLenum internal_format, GLenum data_type){
    createTexture(width,height,color_type,internal_format,data_type,nullptr);
}

void raw_Texture::resize(int width, int height)
{
    this->width = width;
    this->height = height;
    uploadData(nullptr);
    assert_no_glerror();
}


void raw_Texture::createGlTexture(){
    deleteGlTexture();
    /* init_resources */
    glGenTextures(1, &id);
    glBindTexture(target, id);
    setDefaultParameters();
    glBindTexture(target, 0);
    assert_no_glerror();
}

void raw_Texture::deleteGlTexture()
{
    if(id != 0){
        glDeleteTextures(1,&id);
        id = 0;
    }
}


void raw_Texture::uploadData(const GLubyte* data ){
    bind();
    glTexImage2D(target, // target
                 0,  // level, 0 = base, no minimap,
                 static_cast<GLint>(internal_format), // internalformat
                 width,  // width
                 height,  // height
                 0,
                 color_type,  // format
                 data_type, // type
                 data);
    assert_no_glerror();
    unbind();
}

void raw_Texture::uploadSubImage(int x, int y, int width, int height,GLubyte* data ){
    bind();
    glTexSubImage2D(target, 0, x, y, width, height, color_type, data_type, data);
    assert_no_glerror();
    unbind();
}


bool raw_Texture::downloadFromGl(GLubyte* data ){
    if(id==0){
        return false;
    }

    bind();
    glGetTexImage(target,
                  0,
                  color_type,
                  data_type,
                  data);
    assert_no_glerror();
    unbind();
    return true;
}

void raw_Texture::bind(){
    glBindTexture(target, id);
    assert_no_glerror();
}

void raw_Texture::bind(int location){
    glActiveTexture(GL_TEXTURE0+location);
    assert_no_glerror();
    bind();
}


void raw_Texture::unbind(){
    glBindTexture(target, 0);
    assert_no_glerror();
}

void raw_Texture::bindImageTexture(GLuint imageUnit, GLint level, GLboolean layered, GLint layer, GLenum access, GLenum format)
{
    glBindImageTexture(imageUnit,id , level, layered, layer, access, format);
}

void raw_Texture::bindImageTexture(GLuint imageUnit, GLint level, GLboolean layered, GLint layer, GLenum access)
{
    bindImageTexture(imageUnit,level,layered,layer,access,internal_format);
}

void raw_Texture::bindImageTexture(GLuint imageUnit, GLenum access)
{
    bindImageTexture(imageUnit,0,GL_FALSE,0,access);
}



void raw_Texture::setWrap(GLenum param){
    bind();
    glTexParameteri(target, GL_TEXTURE_WRAP_S, static_cast<GLint>(param));
    glTexParameteri(target, GL_TEXTURE_WRAP_T, static_cast<GLint>(param));
    unbind();
    assert_no_glerror();
}
void raw_Texture::setFiltering(GLenum param){
    bind();
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, static_cast<GLint>(param));
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, static_cast<GLint>(param));
    unbind();
    assert_no_glerror();
}

void raw_Texture::setParameter(GLenum name, GLenum param){
    bind();
    glTexParameteri(target, name, static_cast<GLint>(param));
    unbind();
    assert_no_glerror();
}

void raw_Texture::generateMipmaps()
{
    bind();
    glGenerateMipmap(target);
    unbind();

    setParameter(GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_LINEAR);
    assert_no_glerror();
}

void raw_Texture::setBorderColor(vec4 color)
{
    bind();
    glTexParameterfv(target,GL_TEXTURE_BORDER_COLOR,&color[0]);
    unbind();
    assert_no_glerror();
}



//============= Required state: SPECIFIED =============

int raw_Texture::bytesPerChannel(){
    int bytes_per_channel = 0;
    switch(data_type){
    case GL_UNSIGNED_BYTE:
        bytes_per_channel = 1;
        break;
    case GL_UNSIGNED_SHORT:
        bytes_per_channel = 2;
        break;
    case GL_UNSIGNED_INT:
        bytes_per_channel = 4;
        break;
    default:
        break;
    }
    return bytes_per_channel;
}

int raw_Texture::colorChannels(){
    int channels = 0;
    switch(color_type){
    case GL_RED:
        channels = 1;
        break;
    case GL_RGB:
        channels = 3;
        break;
    case GL_RGBA:
        channels = 4;
        break;
    default:
        break;
    }
    return channels;
}

int raw_Texture::bytesPerPixel(){
    return bytesPerChannel()*colorChannels();
}



void raw_Texture::setFormat(const ImageFormat &format){
    internal_format = format.getGlInternalFormat();
    color_type = format.getGlFormat();
    data_type = format.getGlType();
}

void raw_Texture::setFormat(const Image &image)
{
    setFormat(image.Format());
    width = image.width;
    height = image.height;
}

}
