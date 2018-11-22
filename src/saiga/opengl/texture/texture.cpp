/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/texture/texture.h"

namespace Saiga
{
void basic_Texture_2D::setDefaultParameters()
{
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, static_cast<GLint>(GL_LINEAR));
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, static_cast<GLint>(GL_LINEAR));
    glTexParameteri(target, GL_TEXTURE_WRAP_S, static_cast<GLint>(GL_CLAMP_TO_EDGE));
    glTexParameteri(target, GL_TEXTURE_WRAP_T, static_cast<GLint>(GL_CLAMP_TO_EDGE));
}

bool basic_Texture_2D::fromImage(const Image& img, bool srgb, bool flipY)
{
    setFormat(img.type, srgb);
    width  = img.width;
    height = img.height;
    createGlTexture();

    if (flipY)
    {
        std::vector<char> data(img.pitchBytes * img.h);
        for (int i = 0; i < img.h; ++i)
        {
            memcpy(&data[i * img.pitchBytes], img.rowPtr(img.h - i - 1), img.pitchBytes);
        }
        uploadData(data.data());
    }
    else
    {
        uploadData(img.data());
    }

    return true;
}

//====================================================================================


void multisampled_Texture_2D::setDefaultParameters()
{
    //    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, static_cast<GLint>(GL_LINEAR));
    //    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, static_cast<GLint>(GL_LINEAR));
    //    glTexParameteri(target, GL_TEXTURE_WRAP_S, static_cast<GLint>(GL_CLAMP_TO_EDGE));
    //    glTexParameteri(target, GL_TEXTURE_WRAP_T,static_cast<GLint>( GL_CLAMP_TO_EDGE));
}


void multisampled_Texture_2D::uploadData(const void* data)
{
    bind();
    glTexImage2DMultisample(target, samples, internal_format, width, height,
                            GL_TRUE  // fixedsamplelocations
    );
    unbind();
}

}  // namespace Saiga
