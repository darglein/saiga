/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once


#include "saiga/image/image.h"
#include "saiga/opengl/opengl.h"

namespace Saiga
{
class SAIGA_GLOBAL raw_Texture
{
   protected:
    GLuint id = 0;
    GLenum target;
    int width, height;
    GLenum internal_format;
    GLenum color_type, data_type;

   public:
    raw_Texture(GLenum target) : target(target) {}
    virtual ~raw_Texture();

    raw_Texture(raw_Texture const&) = delete;
    raw_Texture& operator=(raw_Texture const&) = delete;

    void createTexture(int width, int height, GLenum color_type, GLenum internal_format, GLenum data_type);
    void createTexture(int width, int height, GLenum color_type, GLenum internal_format, GLenum data_type,
                       const void* data);
    void createEmptyTexture(int width, int height, GLenum color_type, GLenum internal_format, GLenum data_type);

    /**
     * Resizes the texture.
     * The old texture data is lost and the new texture is again uninitialized.
     */
    void resize(int width, int height);

    void createGlTexture();
    void deleteGlTexture();
    virtual void setDefaultParameters() = 0;


    bool downloadFromGl(void* data);

    virtual void uploadData(const void* data);

    void uploadSubImage(int x, int y, int width, int height, void* data);


    void bind();
    void bind(int location);
    void unbind();

    /**
     * Directly maps to glBindImageTexture.
     * See here https://www.opengl.org/sdk/docs/man/html/glBindImageTexture[0]html
     *
     * The OpenGL 4.2+ way of binding textures...
     *
     * If you want to read or write textures from Compute Shaders these functions have to be used instead of the normal
     * bind(...).
     *
     * In Code:
     *
     *   tex.bindImageTexture(3,GL_WRITE_ONLY);
     *
     * In Shader:
     *
     *  layout(binding=3, rgba8) uniform image2D destTex;
     *
     */

    void bindImageTexture(GLuint imageUnit, GLint level, GLboolean layered, GLint layer, GLenum access, GLenum format);
    void bindImageTexture(GLuint imageUnit, GLint level, GLboolean layered, GLint layer, GLenum access);
    void bindImageTexture(GLuint imageUnit, GLenum access);


    int getWidth() { return width; }
    int getHeight() { return height; }
    GLuint getId() { return id; }
    GLenum getTarget() { return target; }


    int bytesPerPixel();
    int bytesPerChannel();
    int colorChannels();

    //    void setFormat(const Image &img);
    void setFormat(ImageType type, bool srgb = false);
    //    void setFormat(const ImageFormat &format);

    void setBorderColor(vec4 color);

    /**
     * GL_CLAMP_TO_EDGE,
     * GL_CLAMP_TO_BORDER,
     * GL_MIRRORED_REPEAT,
     * GL_REPEAT, or
     * GL_MIRROR_CLAMP_TO_EDGE
     */
    void setWrap(GLenum param);

    /**
     * GL_NEAREST
     * GL_LINEAR
     * GL_NEAREST_MIPMAP_NEAREST
     * GL_LINEAR_MIPMAP_NEAREST
     * GL_NEAREST_MIPMAP_LINEAR
     * GL_LINEAR_MIPMAP_LINEAR
     */

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

}  // namespace Saiga
