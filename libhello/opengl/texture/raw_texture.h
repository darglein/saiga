#pragma once

#include "libhello/opengl/texture/image.h"

class raw_Texture{

protected:
    GLuint id = 0;
    const GLenum target;
    int width,height;
    int internal_format,color_type,data_type;

    int channel_depth=0; //number of bytes per channel
    int channels=0; //number of channels. example: RGB has 3 channels
public:
    raw_Texture(GLenum target):target(target){}
    virtual ~raw_Texture();

    void createTexture(int width, int height, int color_type, int internal_format, int data_type);
    void createTexture(int width, int height, int color_type, int internal_format, int data_type,GLubyte* data );
    void createEmptyTexture(int width, int height, int color_type, int internal_format, int data_type);

    void createGlTexture();
    virtual void setDefaultParameters() = 0;

    bool isValid();
    bool isSpecified();


    //============= Required state: VALID =============

    bool downloadFromGl(GLubyte *data);

    virtual void uploadData(GLubyte* data);

    void uploadSubImage(int x, int y, int width, int height,GLubyte* data );
    //only works for 3d textures!!
    void uploadSubImage(int x, int y, int z, int width, int height, int depth, GLubyte *data);

    virtual void bind();
    virtual void bind(int location);
    virtual void unbind();

    void setWrap(GLint param);
    void setFiltering(GLint param);

    int getWidth(){return width;}
    int getHeight(){return height;}
    GLuint getId(){return id;}
    GLenum getTarget(){return target;}

    void specify(int channel_depth, int channels, int srgb);

    //============= Required state: SPECIFIED =============

    GLubyte* downloadFromGl();
    int bytesPerPixel();
    int bytesPerChannel();
    int colorChannels();

    void setFormat(const Image &img);



    void setParameter(GLenum name, GLint param);
};

