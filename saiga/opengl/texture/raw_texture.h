#pragma once

#include "saiga/util/glm.h"
#include "saiga/opengl/texture/image.h"

class SAIGA_GLOBAL raw_Texture{

protected:
    GLuint id = 0;
    const GLenum target;
    int width,height;
    GLenum internal_format;
    GLenum color_type,data_type;

    int channel_depth=0; //number of bytes per channel
    int channels=0; //number of channels. example: RGB has 3 channels
public:
    raw_Texture(GLenum target):target(target){}
    virtual ~raw_Texture();

    void createTexture(int width, int height, GLenum color_type, GLenum internal_format, GLenum data_type);
    void createTexture(int width, int height, GLenum color_type, GLenum internal_format, GLenum data_type,GLubyte* data );
    void createEmptyTexture(int width, int height, GLenum color_type, GLenum internal_format, GLenum data_type);

    /**
     * Resizes the texture.
     * The old texture data is lost and the new texture is again uninitialized.
     */
    void resize(int width, int height);

    void createGlTexture();
    virtual void setDefaultParameters() = 0;

    bool isValid();
    bool isSpecified();


    bool downloadFromGl(GLubyte *data);

    virtual void uploadData(GLubyte* data);

    void uploadSubImage(int x, int y, int width, int height,GLubyte* data );
    //only works for 3d textures!!
    void uploadSubImage(int x, int y, int z, int width, int height, int depth, GLubyte *data);

    virtual void bind();
    virtual void bind(int location);
    virtual void unbind();

    /**
     * Directly maps to glBindImageTexture.
     * See here https://www.opengl.org/sdk/docs/man/html/glBindImageTexture.xhtml
     *
     * The OpenGL 4.2+ way of binding textures...
     *
     * If you want to read or write textures from Compute Shaders these functions have to be used instead of the normal bind(...).
     */

    void bindImageTexture(GLuint imageUnit, GLint level, GLboolean layered, GLint layer, GLenum access, GLenum format);
    void bindImageTexture(GLuint imageUnit, GLint level, GLboolean layered, GLint layer, GLenum access);
    void bindImageTexture(GLuint imageUnit, GLenum access);


    int getWidth(){return width;}
    int getHeight(){return height;}
    GLuint getId(){return id;}
    GLenum getTarget(){return target;}

    void specify(int channel_depth, int channels, int srgb);


    GLubyte* downloadFromGl();
    int bytesPerPixel();
    int bytesPerChannel();
    int colorChannels();

    void setFormat(const Image &img);


    void setBorderColor(vec4 color);
    void setWrap(GLenum param);
    void setFiltering(GLenum param);
    void setParameter(GLenum name, GLenum param);

    /*
     *
     * Mipmap generation replaces texel image levels levelbase+1 through q with images derived from the levelbase image,
     * regardless of their previous contents. All other mimap images,
     * including the levelbase+1 image, are left unchanged by this computation.

     * The internal formats of the derived mipmap images all match those of the levelbase image.
     * The contents of the derived images are computed by repeated, filtered reduction of the levelbase+1 image.
     * For one- and two-dimensional array and cube map array textures, each layer is filtered independently.
     *
     * Cube map and array textures are accepted only if the GL version is 4.0 or higher.
     */

    void generateMipmaps();
};

