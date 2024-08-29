#include "libtiff_wrapper.h"

#ifdef SAIGA_USE_LIBTIFF

#    include "tiffio.h"
#    include "tiffio.hxx"

#    include <iostream>

namespace Saiga
{
bool loadImageLibTiff(const std::string& path, Image& img)
{
    TIFFSetWarningHandler(nullptr);
    TIFFSetWarningHandlerExt(nullptr);

    TIFF* tif = TIFFOpen(path.c_str(), "r");

    if (!tif) return false;

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

    // size_t buffer_size = TIFFScanlineSize(tif);
    // SAIGA_ASSERT(buffer_size == img.pitchBytes);

    for (uint32_t row = 0; row < height; ++row)
    {
        TIFFReadScanline(tif, img.rowPtr(row), row);
    }


    TIFFClose(tif);

    return true;
}
bool saveImageLibTiff(const std::string& path, const Image& img)
{
    TIFFSetWarningHandler(nullptr);
    TIFFSetWarningHandlerExt(nullptr);

    TIFF* tif = TIFFOpen(path.c_str(), "w");
    if (!tif) return false;

    uint32_t width  = img.width;
    uint32_t height = img.height;

    // Get the image width and height
    if (!TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width))
    {
        return false;
    }
    if (!TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height))
    {
        return false;
    }


    uint16_t bitspersample = 0, sample_format = SAMPLEFORMAT_UINT;
    uint16_t samples = 1;
    switch (img.type)
    {
        case ImageType::UC1:
            sample_format = SAMPLEFORMAT_UINT;
            bitspersample = 8;
            break;
        case ImageType::US1:
            sample_format = SAMPLEFORMAT_UINT;
            bitspersample = 16;
            break;
        case ImageType::F1:
            sample_format = SAMPLEFORMAT_IEEEFP;
            bitspersample = 32;
            break;
        default:
            return false;
    }


    if (!TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, bitspersample))
    {
        return false;
    }

    // These tags might actually be not present. In these cases the default values are used.
    if (!TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, sample_format))
    {
        return false;
    }
    if (!TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, samples))
    {
        return false;
    }


    for (uint32_t row = 0; row < height; ++row)
    {
        TIFFWriteScanline(tif, const_cast<void*>(img.rowPtr(row)), row);
    }

    TIFFClose(tif);

    return true;
}

struct membuf : std::streambuf
{
    membuf(char const* base, size_t size)
    {
        char* p(const_cast<char*>(base));
        this->setg(p, p, p + size);
    }

    pos_type seekoff(off_type off, std::ios_base::seekdir dir, std::ios_base::openmode which = std::ios_base::in) override
    {
        if (dir == std::ios_base::cur)
            gbump(off);
        else if (dir == std::ios_base::end)
            setg(eback(), egptr() + off, egptr());
        else if (dir == std::ios_base::beg)
            setg(eback(), eback() + off, egptr());
        return gptr() - eback();
    }

    pos_type seekpos(pos_type sp, std::ios_base::openmode which) override
    {
        return seekoff(sp - pos_type(off_type(0)), std::ios_base::beg, which);
    }
};
struct imemstream : virtual membuf, std::istream
{
    imemstream(char const* base, size_t size) : membuf(base, size), std::istream(static_cast<std::streambuf*>(this)) {}
};

bool loadImageFromMemoryLibTiff(const void* data, size_t size, Image& img)
{
    TIFFSetWarningHandler(nullptr);
    TIFFSetWarningHandlerExt(nullptr);


    imemstream strm((const char*)data, size);
    // TIFF* tif = TIFFOpen(path.c_str(), "r");
    TIFF* tif = TIFFStreamOpen("balbla", &strm);

    if (!tif) return false;

    uint32_t width, height;
    uint16_t bitspersample = 0, sample_format = SAMPLEFORMAT_UINT, samples = 1;

    // Get the image width and height
    if (!TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width))
    {
        return false;
    }
    if (!TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height))
    {
        return false;
    }
    if (!TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitspersample))
    {
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
        samples = 1;
    }

    type = (ImageType)((int)type + samples - 1);
    img.create(height, width, type);

    // size_t buffer_size = TIFFScanlineSize(tif);
    // SAIGA_ASSERT(buffer_size == img.pitchBytes);

    for (uint32_t row = 0; row < height; ++row)
    {
        TIFFReadScanline(tif, img.rowPtr(row), row);
    }


    TIFFClose(tif);

    return true;
}


}  // namespace Saiga

#endif
