#include "libtiff_wrapper.h"

#ifdef SAIGA_USE_LIBTIFF

#    include "tiffio.h"

#    include <iostream>

namespace Saiga
{
bool loadImageLibTiff(const std::string& path, Image& img)
{
    TIFFSetWarningHandler(nullptr);
    TIFFSetWarningHandlerExt(nullptr);

    TIFF* tif = TIFFOpen(path.c_str(), "r");

    if (tif) 
    {
        uint32_t width, height;
        uint16_t bitspersample = 0, sample_format = SAMPLEFORMAT_UINT, samples = 1;

        // Get the image width and height
        if (!TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width))
        {
            std::cerr << "The file '" << path << "' has no 'TIFFTAG_IMAGEWIDTH'. The file is probably corrupted\n";
            return false;
        }
        if (!TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height))
        {
            std::cerr << "The file '" << path << "' has no 'TIFFTAG_IMAGELENGTH'. The file is probably corrupted\n";
            return false;
        }
        if (!TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitspersample)) 
        {
            std::cerr << "The file '" << path << "' has no 'TIFFTAG_BITSPERSAMPLE'. The file is probably corrupted\n";
            return false;
        }

        // These tags might actually be not present. In these cases the default values are used.
        if (!TIFFGetField(tif, TIFFTAG_SAMPLEFORMAT, &sample_format))
        {
            sample_format = SAMPLEFORMAT_UINT;
        }
        if (!TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samples))
        {
            samples = 1;
        }


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
            std::cout << "Image '" << path << "' has more than one channel. Only loading first channel\n";
            samples = 1;
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
