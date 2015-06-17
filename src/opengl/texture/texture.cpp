
#include "opengl/texture/texture.h"


Texture* TextureLoader::loadFromFile(const std::string &path){

    bool erg;
    Texture* text = new Texture();

//    PNG::Image img;
//    erg = PNG::readPNG( &img,path);
//    cout<<"loading "<<path<<endl;

    fipImage fipimg;
    erg = fipimg.load(path.c_str());


    if (erg){
        Image im;
        im.convertFrom(fipimg);
        im.srgb = true;
        erg = text->fromImage(im);
    }

    if(erg){
        return text;
    }else{
        delete text;
    }



    return nullptr;
}

void basic_Texture_2D::setDefaultParameters(){
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, static_cast<GLint>(GL_LINEAR));
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, static_cast<GLint>(GL_LINEAR));
    glTexParameteri(target, GL_TEXTURE_WRAP_S, static_cast<GLint>(GL_CLAMP_TO_EDGE));
    glTexParameteri(target, GL_TEXTURE_WRAP_T,static_cast<GLint>( GL_CLAMP_TO_EDGE));

//    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
}

bool basic_Texture_2D::fromImage(Image &img){


    setFormat(img);

    createGlTexture();
    uploadData(img.data);
    return true;
}
