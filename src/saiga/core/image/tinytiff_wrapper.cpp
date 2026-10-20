#include "tinytiff_wrapper.h"

#ifdef SAIGA_USE_TINYTIFF

#    include <cstring>
#    include <filesystem>
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
        // For interleaved samples, TinyTIFF copies as many samples as the strip byte counts declare without checking
        // the size of the output buffer, so a corrupt file overflows it. The planar configuration is not exposed, so
        // multi channel images are rejected.
        std::cerr << "The file '" << path << "' has " << samples << " samples per pixel, which TinyTIFF cannot load\n";
        return false;
    }

    // TinyTIFF writes the sample densely (without row padding), but the rows of img are aligned to 4 bytes.
    const size_t row_bytes = size_t(width) * (bitspersample / 8);
    // TinyTIFF only reads uncompressed images, so the pixel data must fit into the file. This rejects corrupt
    // headers before they cause a huge allocation.
    std::error_code ec;
    const auto file_size = std::filesystem::file_size(path, ec);
    if (ec || uint64_t(row_bytes) * height > file_size)
    {
        std::cerr << "The file '" << path << "' is too small for an image of size " << width << "x" << height << "\n";
        return false;
    }
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
