#pragma once

#include "saiga/util/glm.h"
#include "saiga/opengl/opengl.h"
#include "saiga/image/glImageFormatMap.h"

namespace Saiga {

class SAIGA_GLOBAL ImageFormat{
private:
    int channels;
    int bitDepth;
    ImageElementFormat elementFormat;
    bool srgb;

public:
    //the default format is RGBA 8bit normalized
    ImageFormat(int channels = 4, int bitDepth = 8, ImageElementFormat elementFormat = ImageElementFormat::UnsignedNormalized, bool srgb = false);

    //basic getters and setters
    int getChannels() const;
    void setChannels(int value);
    int getBitDepth() const;
    void setBitDepth(int value);
    ImageElementFormat getElementFormat() const;
    void setElementFormat(const ImageElementFormat &value);
    bool getSrgb() const;
    void setSrgb(bool value);

    int bytesPerChannel();
    int bytesPerPixel();
    int bitsPerPixel();

    //match to the parameters of glTexImage2D(...)
    //https://www.opengl.org/sdk/docs/man/html/glTexImage2D.xhtml
    GLenum getGlInternalFormat() const;
    GLenum getGlFormat() const;
    GLenum getGlType() const;
};


SAIGA_GLOBAL std::ostream& operator<<(std::ostream& os, const ImageFormat& f);

}
