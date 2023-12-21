#include "tinytiff_wrapper.h"

#ifdef SAIGA_USE_TINYTIFF

#    include <iostream>
#    include <tinytiffreader.h>

namespace Saiga
{
bool loadImageTinyTiff(const std::string& path, Image& img)
{
    TinyTIFFReaderFile* tiffr = NULL;
    tiffr                     = TinyTIFFReader_open(path.c_str());
    if (!tiffr)
    {
        std::cout << "    ERROR reading (not existent, not accessible or no TIFF file)\n";
        return false;
    }

    if (TinyTIFFReader_wasError(tiffr))
    {
        std::cout << "   ERROR:" << TinyTIFFReader_getLastError(tiffr) << "\n";
        return false;
    }

    //std::cout << "    ImageDescription:\n" << TinyTIFFReader_getImageDescription(tiffr) << "\n";
    uint32_t frames = TinyTIFFReader_countFrames(tiffr);
    //std::cout << "    frames: " << frames << "\n";
    if (TinyTIFFReader_wasError(tiffr))
    {
        std::cout << "   ERROR:" << TinyTIFFReader_getLastError(tiffr) << "\n";
        return false;
    }

    SAIGA_ASSERT(frames == 1);

    const uint32_t width         = TinyTIFFReader_getWidth(tiffr);
    const uint32_t height        = TinyTIFFReader_getHeight(tiffr);
    const uint16_t samples       = TinyTIFFReader_getSamplesPerPixel(tiffr);
    const uint16_t bitspersample = TinyTIFFReader_getBitsPerSample(tiffr, 0);

    const uint16_t sample_format = TinyTIFFReader_getSampleFormat(tiffr);

    ImageType type = ImageType::UC1;
    if (sample_format == TINYTIFF_SAMPLEFORMAT_FLOAT)
    {
        type = ImageType::F1;
    }
    else if (sample_format == TINYTIFF_SAMPLEFORMAT_UINT)
    {
        type = (bitspersample == 8) ? ImageType::UC1 : (bitspersample == 16) ? ImageType::US1 : ImageType::UI1;
    }
    else if (sample_format == TINYTIFF_SAMPLEFORMAT_INT)
    {
        type = (bitspersample == 8) ? ImageType::C1 : (bitspersample == 16) ? ImageType::S1 : ImageType::I1;
    }

    type = (ImageType)((int)type + samples - 1);
    img.create(height, width, type);

    bool ok                      = true;
    //std::cout << "    size of frame " << frame << ": " << width << "x" << height << "\n";
    //std::cout << "    each pixel has " << samples << " samples with " << bitspersample << " bits each\n";
    if (ok)
    {
        for (uint16_t sample = 0; sample < samples; sample++)
        {
            // read the sample
            TinyTIFFReader_getSampleData(tiffr, img.rowPtr(0), sample);
            if (TinyTIFFReader_wasError(tiffr))
            {
                ok = false;
                std::cout << "   ERROR:" << TinyTIFFReader_getLastError(tiffr) << "\n";
                break;
            }
        }
    }

    TinyTIFFReader_close(tiffr);

    return ok;
}
}  // namespace Saiga

#endif
