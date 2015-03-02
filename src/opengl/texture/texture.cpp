
#include "opengl/texture/texture.h"


Texture* TextureLoader::loadFromFile(const std::string &path){

    bool erg;
    Texture* text = new Texture();

//    PNG::Image img;
//    erg = PNG::readPNG( &img,path);
    cout<<"loading "<<path<<endl;

    fipImage fipimg;
    erg = fipimg.load(path.c_str());


    if (erg){
        Image im;
        im.convertFrom(fipimg);
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
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

bool basic_Texture_2D::fromImage(Image &img){


    setFormat(img);

    createGlTexture();
    uploadData(img.data);
    return true;
}
