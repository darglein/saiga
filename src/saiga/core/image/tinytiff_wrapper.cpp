#include "tinytiff_wrapper.h"

#ifdef SAIGA_USE_TINYTIFF

#    include <cstring>
#    include <iostream>
#    include <limits>
#    include <memory>
#    include <tinytiffreader.h>
#    include <vector>

namespace Saiga
{
// Loads the first channel of the first frame, like the libtiff wrapper.
bool loadImageTinyTiff(const std::string& path, Image& img)
{
    TinyTIFFReaderFile* tiffr = TinyTIFFReader_open(path.c_str());
    if (!tiffr)
    {
        std::cerr << "Could not open the tiff file '" << path << "'\n";
        return false;
    }
    // Closes the reader on every return path.
    std::unique_ptr<TinyTIFFReaderFile, decltype(&TinyTIFFReader_close)> guard(tiffr, &TinyTIFFReader_close);

    if (TinyTIFFReader_wasError(tiffr))
    {
        std::cerr << "Error reading '" << path << "': " << TinyTIFFReader_getLastError(tiffr) << "\n";
        return false;
    }

    const uint32_t width         = TinyTIFFReader_getWidth(tiffr);
    const uint32_t height        = TinyTIFFReader_getHeight(tiffr);
    const uint16_t samples       = TinyTIFFReader_getSamplesPerPixel(tiffr);
    const uint16_t bitspersample = TinyTIFFReader_getBitsPerSample(tiffr, 0);
    const uint16_t sample_format = TinyTIFFReader_getSampleFormat(tiffr);

    ImageType type = TYPE_UNKNOWN;
    if (sample_format == TINYTIFF_SAMPLEFORMAT_FLOAT)
    {
        type = (bitspersample == 32) ? ImageType::F1 : (bitspersample == 64) ? ImageType::D1 : TYPE_UNKNOWN;
    }
    else if (sample_format == TINYTIFF_SAMPLEFORMAT_UINT)
    {
        type = (bitspersample == 8)    ? ImageType::UC1
               : (bitspersample == 16) ? ImageType::US1
               : (bitspersample == 32) ? ImageType::UI1
                                       : TYPE_UNKNOWN;
    }
    else if (sample_format == TINYTIFF_SAMPLEFORMAT_INT)
    {
        type = (bitspersample == 8)    ? ImageType::C1
               : (bitspersample == 16) ? ImageType::S1
               : (bitspersample == 32) ? ImageType::I1
                                       : TYPE_UNKNOWN;
    }
    if (type == TYPE_UNKNOWN)
    {
        std::cerr << "The file '" << path << "' has an unsupported pixel format (" << bitspersample
                  << " bits per sample, sample format " << sample_format << ")\n";
        return false;
    }
    if (samples == 0)
    {
        std::cerr << "The file '" << path << "' has 0 samples per pixel\n";
        return false;
    }
    if (width == 0 || height == 0 || width > uint32_t(std::numeric_limits<int>::max() / 8) ||
        height > uint32_t(std::numeric_limits<int>::max()))
    {
        std::cerr << "The file '" << path << "' has an invalid size " << width << "x" << height << "\n";
        return false;
    }
    if (samples != 1)
    {
        std::cout << "Image '" << path << "' has more than one channel. Only loading first channel\n";
    }

    // TinyTIFF writes the sample densely (without row padding), but the rows of img are aligned to 4 bytes.
    const size_t row_bytes = size_t(width) * (bitspersample / 8);
    std::vector<unsigned char> buffer(row_bytes * height);
    if (!TinyTIFFReader_getSampleData(tiffr, buffer.data(), 0) || TinyTIFFReader_wasError(tiffr))
    {
        std::cerr << "Error reading '" << path << "': " << TinyTIFFReader_getLastError(tiffr) << "\n";
        return false;
    }

    img.create(height, width, type);
    for (uint32_t y = 0; y < height; ++y)
    {
        std::memcpy(img.rowPtr(y), buffer.data() + y * row_bytes, row_bytes);
    }
    return true;
}
}  // namespace Saiga

#endif
