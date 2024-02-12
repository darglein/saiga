#include "libtiff_wrapper.h"

#ifdef SAIGA_USE_LIBTIFF

#    include "tiffio.h"

#    include <iostream>

namespace Saiga
{
bool loadImageLibTiff(const std::string& path, Image& img)
{
    TIFF* tif = TIFFOpen(path.c_str(), "r");

    if (tif) 
    {
        uint32_t width, height;
        uint16_t bitspersample, sample_format, samples;

        // Get the image width and height
        TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
        TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);

        TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitspersample);
        TIFFGetField(tif, TIFFTAG_SAMPLEFORMAT, &sample_format);
        TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samples);


        ImageType type = ImageType::UC1;
        if (sample_format == SAMPLEFORMAT_IEEEFP)
        {
            type = ImageType::F1;
        }
        else if (sample_format == SAMPLEFORMAT_UINT)
        {
            type = (bitspersample == 8) ? ImageType::UC1 : (bitspersample == 16) ? ImageType::US1 : ImageType::UI1;
        }
        else if (sample_format == SAMPLEFORMAT_INT)
        {
            type = (bitspersample == 8) ? ImageType::C1 : (bitspersample == 16) ? ImageType::S1 : ImageType::I1;
        }

        if (samples != 1)
        {
            throw std::runtime_error("We currently only support 1 channel tifs.");
        }

        type = (ImageType)((int)type + samples - 1);
        img.create(height, width, type);

        //size_t buffer_size = TIFFScanlineSize(tif);
        //SAIGA_ASSERT(buffer_size == img.pitchBytes);

        for (uint32_t row = 0; row < height; ++row)
        {
            TIFFReadScanline(tif, img.rowPtr(row), row);
        }


        TIFFClose(tif);

        return true;
    }
    
    return false;
}
}  // namespace Saiga

#endif
