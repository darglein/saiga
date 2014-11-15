#ifndef TEXTURE_H
#define TEXTURE_H

#include <SDL2/SDL.h>
#include <GL/glew.h>
#include <SDL2/SDL_opengl.h>
#include <GL/glu.h>


#include <vector>
#include <string>
#include <iostream>
#include "libhello/util/loader.h"
#include "libhello/util/error.h"
#include "libhello/util/png_wrapper.h"

using std::string;
using std::cout;
using std::endl;

class Texture{

public:
    string name;
    GLuint id = 0;
    int width,height;
    GLubyte* data = NULL;
    int internal_format,color_type,data_type;

public:


    Texture(){}
    Texture(const string &name);
    virtual ~Texture();
    void createGlTexture();
    bool downloadFromGl();

    void createTexture(int width, int height, int color_type, int internal_format, int data_type,GLubyte* data );
    void createEmptyTexture(int width, int height, int color_type, int internal_format, int data_type);

    void uploadSubImage(int x, int y, int width, int height,GLubyte* data );

    void bind();
    void bind(int location);
    void unBind();

    bool toPNG(PNG::Image *out_img);
    bool fromPNG(PNG::Image *img);

    int bytesPerPixel();
    int bytesPerChannel();
    int colorChannels();

//    void unpackPixelAlignment(){glBindTexture(GL_TEXTURE_2D, id);glPixelStorei(GL_UNPACK_ALIGNMENT, 1);}


};

class TextureLoader : public Loader<Texture>{
public:
    Texture* loadFromFile(const std::string &name);
};

#endif // TEXTURE_H
