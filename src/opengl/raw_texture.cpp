
#include "opengl/raw_texture.h"



raw_Texture::~raw_Texture(){
    if(id != 0)
        glDeleteTextures(1,&id);
}

void raw_Texture::createTexture(int width, int height, int color_type, int internal_format, int data_type){
    this->width = width;
    this->height = height;
    this->color_type = color_type;
    this->data_type = data_type;
    this->internal_format = internal_format;
    createGlTexture();
}

void raw_Texture::createTexture(int width, int height, int color_type, int internal_format, int data_type, GLubyte* data){
    createTexture(width,height,color_type, internal_format,  data_type);
    uploadData(data);
}


void raw_Texture::createEmptyTexture(int width, int height, int color_type, int internal_format, int data_type){
    createTexture(width,height,color_type,internal_format,data_type,NULL);
}


void raw_Texture::createGlTexture(){
    /* init_resources */
    glGenTextures(1, &id);
    glBindTexture(target, id);
    setDefaultParameters();
    glBindTexture(target, 0);

    Error::quitWhenError("createGlTexture()");
}

bool raw_Texture::isValid(){
    return id!=0;
}

bool raw_Texture::isSpecified(){
    return channel_depth!=0 && channels!=0;
}

//============= Required state: VALID =============


void raw_Texture::uploadData(GLubyte* data ){
    bind();
    glTexImage2D(target, // target
                 0,  // level, 0 = base, no minimap,
                 internal_format, // internalformat
                 width,  // width
                 height,  // height
                 0,  // border, always 0 in OpenGL ES
                 color_type,  // format
                 data_type, // type
                 data);
    unbind();
}

void raw_Texture::uploadSubImage(int x, int y, int width, int height,GLubyte* data ){
    bind();
    glTexSubImage2D(target, 0, x, y, width, height, color_type, data_type, data);
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
    unbind();
    return true;
}

void raw_Texture::bind(){
    glBindTexture(target, id);
}

void raw_Texture::bind(int location){
    glActiveTexture(GL_TEXTURE0+location);
    bind();
}


void raw_Texture::unbind(){
    glBindTexture(target, 0);
}

void raw_Texture::setWrap(GLint param){
    bind();
    glTexParameteri(target, GL_TEXTURE_WRAP_S, param);
    glTexParameteri(target, GL_TEXTURE_WRAP_T, param);
    unbind();
}
void raw_Texture::setFiltering(GLint param){
    bind();
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, param);
    unbind();
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
    }
    return channels;
}

int raw_Texture::bytesPerPixel(){
    return bytesPerChannel()*colorChannels();
}


bool raw_Texture::toPNG(PNG::Image* out_img){
//    if(data==nullptr){
//        cout<<"Can't create PNG::Image: This texture has no data."<<endl;
//        return false;
//    }

//    out_img->data = data;
    out_img->width = width;
    out_img->height = height;

    switch(data_type){
    case GL_UNSIGNED_BYTE:
        out_img->bit_depth = 8;
        break;
    case GL_UNSIGNED_SHORT:
        out_img->bit_depth = 16;
        break;
    default:
        return false;
    }
    switch(color_type){

    case GL_RED:
        out_img->color_type = PNG_COLOR_TYPE_GRAY;
        break;
    case GL_RGB:
        out_img->color_type = PNG_COLOR_TYPE_RGB;
        break;
    case GL_RGBA:
        out_img->color_type = PNG_COLOR_TYPE_RGB_ALPHA ;
        break;
    default:
        return false;
    }
    return true;
}

bool raw_Texture::fromPNG(PNG::Image *img){
//    data = img->data;
    width = img->width;
    height = img->height;


        switch(img->color_type){
        case PNG_COLOR_TYPE_GRAY:
            color_type = GL_RED;
            break;
        case PNG_COLOR_TYPE_RGB:
            color_type = GL_RGB;
            break;
        case PNG_COLOR_TYPE_RGB_ALPHA:
            color_type = GL_RGBA;
            break;
        default:
            std::cout<<"Image type not supported: "<<img->color_type<<std::endl;
            std::cout<<"Supported types: Grayscale, RGB, RGBA"<<std::endl;
            return false;
        }

        internal_format = color_type;

        switch(img->bit_depth){
        case 8:
            data_type = GL_UNSIGNED_BYTE;
            break;
        case 16:
            data_type = GL_UNSIGNED_SHORT;
            break;
        default:
            std::cout<<"Bit depth not supported: "<<img->bit_depth<<std::endl;
            std::cout<<"Supported bit depths: 8,16"<<std::endl;
            return false;
        }
        createGlTexture();
        uploadData(img->data);
        return true;
}

