/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once


#include "saiga/core/image/image.h"
#include "saiga/opengl/opengl.h"

#include <functional>

namespace Saiga
{
class SAIGA_OPENGL_API TextureBase
{
   public:
    TextureBase(GLenum target) : target(target) {}
    virtual ~TextureBase();

    TextureBase(TextureBase const&)            = delete;
    TextureBase& operator=(TextureBase const&) = delete;

    // Create and destroy the underlying GL types.
    void create();
    void create(GLenum color_type, GLenum internal_format, GLenum data_type);
    void destroy();

    // Downloads the GPU data. Returns false for cube maps, use TextureCube::downloadFace.
    bool download(void* data);
    // Downloads the GPU data converted to the given color format and data type (glGetTexImage)
    bool download(void* data, GLenum format, GLenum type);


    // Bind the texture to the target.
    // In most cases you should use bind(int) see below.
    void bind();

    /**
     * Binds the texture (see above) and also binds it to the active texture 'location'.
     * The typical usecase is:
     *
     * 1. bind shader
     * 2. bind texture to unit X
     * 3. update texture uniform at location L to texture unit X
     */
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


    GLuint getId() { return id; }
    GLenum getTarget() { return target; }


    /**
     * Sets the internal formats according to the saiga image type
     */
    void setFormat(ImageType type, bool srgb = false, bool integer = false);

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
     *
     * Integer textures are not filterable and keep their single level (and filter).
     */

    void generateMipmaps();

    GLenum InternalFormat() const { return internal_format; }

   protected:
    /**
     * Calls upload(ptr) with an unpack state (GL_UNPACK_ALIGNMENT, GL_UNPACK_ROW_LENGTH) that reads 'height' rows of
     * 'width' pixels, which are 'pitch_bytes' apart in 'data'. If no unpack state describes this pitch, ptr points to a
     * packed copy of the rows. The previous unpack state is restored afterwards.
     */
    static void uploadWithPitch(int width, int height, ImageType type, size_t pitch_bytes, const void* data,
                                const std::function<void(const void*)>& upload);

    // GL_LINEAR, or GL_NEAREST for integer textures, which are incomplete (fetches return 0) with linear filtering.
    GLenum defaultFilter() const;

    // The underlying GL texture name and target
    GLuint id = 0;
    GLenum target;

    // Every texture has an internal-format/colortype/datatype
    GLenum internal_format;
    GLenum color_type, data_type;
};


}  // namespace Saiga
