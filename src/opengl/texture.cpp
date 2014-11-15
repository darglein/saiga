
#include "opengl/texture.h"


//----------------------- TEXTURE LIST------------------------------


Texture::Texture(const string &name) : name(name),id(0){

}

Texture::~Texture(){
    if(id != 0)
        glDeleteTextures(1,&id);
}



//-----------------------------------------------------------------------

void Texture::createTexture(int width, int height, int color_type, int internal_format, int data_type, GLubyte* data){
    this->width = width;
    this->height = height;
    this->color_type = color_type;
    this->data_type = data_type;
    this->data = data;
    this->internal_format = internal_format;
    createGlTexture();
}


void Texture::createEmptyTexture(int width, int height, int color_type, int internal_format, int data_type){
    createTexture(width,height,color_type,internal_format,data_type,NULL);
}

void Texture::createGlTexture(){
    /* init_resources */
    glGenTextures(1, &id);
    glBindTexture(GL_TEXTURE_2D, id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); 
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexImage2D(GL_TEXTURE_2D, // target
                 0,  // level, 0 = base, no minimap,
                 internal_format, // internalformat
                 width,  // width
                 height,  // height
                 0,  // border, always 0 in OpenGL ES
                 color_type,  // format
                 data_type, // type
                 data);
    glBindTexture(GL_TEXTURE_2D, 0);

    Error::quitWhenError("createGlTexture()");
}

 void Texture::uploadSubImage(int x, int y, int width, int height,GLubyte* data ){
     bind();
     glTexSubImage2D(GL_TEXTURE_2D, 0, x, y, width, height, color_type, data_type, data);
      unBind();
 }

 bool Texture::downloadFromGl(){
     if(id==0){
         return false;
     }
     if(data)
         delete data;
     data = new GLubyte[bytesPerPixel()*width*height];

     bind();
     glGetTexImage(GL_TEXTURE_2D,
                   0,
                   color_type,
                   data_type,
                   data);
     unBind();
     return true;
 }

void Texture::bind(){
    glBindTexture(GL_TEXTURE_2D, id);
}

void Texture::bind(int location){
    glActiveTexture(GL_TEXTURE0+location);
    bind();
}


void Texture::unBind(){
    glBindTexture(GL_TEXTURE_2D, 0);
}

bool Texture::toPNG(PNG::Image* out_img){
    if(data==NULL){
        cout<<"Can't create PNG::Image: This texture has no data."<<endl;
        return false;
    }

    out_img->data = data;
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

bool Texture::fromPNG(PNG::Image *img){
    data = img->data;
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
        return true;
}

int Texture::bytesPerChannel(){
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

int Texture::colorChannels(){
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

int Texture::bytesPerPixel(){
    return bytesPerChannel()*colorChannels();
}

Texture* TextureLoader::loadFromFile(const std::string &path){

    bool erg;
    Texture* text = new Texture();

    PNG::Image img;
    erg = PNG::readPNG( &img,path);
    if (erg){
        erg = text->fromPNG(&img);
    }

    if(erg){
        text->createGlTexture();
        return text;
    }else{
        delete text;
    }



    return NULL;
}

