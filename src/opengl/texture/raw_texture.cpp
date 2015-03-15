
#include "opengl/texture/raw_texture.h"
#include "libhello/util/error.h"


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
    createTexture(width,height,color_type,internal_format,data_type,nullptr);
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
                 0,
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

void raw_Texture::uploadSubImage(int x, int y, int z, int width, int height , int depth, GLubyte* data ){
    bind();
    glTexSubImage3D(target, 0, x, y, z,width, height,depth, color_type, data_type, data);
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
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, param);
    unbind();
}


int glinternalFormat(int channels, int depth){
    int coffset = channels -1;
    int doffset = 0;
    switch(depth){
    case 8:
        doffset = 0;
        break;
    case 16:
        doffset = 1;
        break;
    case 32:
        doffset = 2;
        break;
    default:
        std::cout<<"Bit depth not supported: "<<depth<<std::endl;
        std::cout<<"Supported bit depths: 8,16,32"<<std::endl;
        return 0;
    }

    static const int iformats[4][3] {
        {GL_R8,GL_R16,GL_R32I}, //1 channel
        {GL_RG8,GL_RG16,GL_RG32I}, //2 channels
        {GL_RGB8,GL_RGB16,GL_RGB32I}, //3 channels
        {GL_RGBA8,GL_RGBA16,GL_RGBA32I} //4 channels
    };

    return iformats[coffset][doffset];



}

void raw_Texture::specify(int channel_depth,int channels){
    switch(channel_depth){
    case 8:
        data_type = GL_UNSIGNED_BYTE;
        break;
    case 16:
        data_type = GL_UNSIGNED_SHORT;
        break;
    case 32:
        data_type = GL_UNSIGNED_INT;
        break;
    default:
        std::cout<<"Bit depth not supported: "<<channel_depth<<std::endl;
        std::cout<<"Supported bit depths: 8,16,32"<<std::endl;
    }
    switch(channels){
    case 1:
        color_type = GL_RED;
        break;
    case 2:
        color_type = GL_RG;
        break;
    case 3:
        color_type = GL_RGB ;
        break;
    case 4:
        color_type = GL_RGBA ;
        break;
    default:
        std::cout<<"Channels not supported: "<<channels<<std::endl;
    }
    internal_format = glinternalFormat(channels,channel_depth);
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




void raw_Texture::setFormat(const Image &img){
    width = img.width;
    height = img.height;

    specify(img.getBitDepth(),img.getChannels());
}

