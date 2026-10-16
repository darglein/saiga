#include "libtiff_wrapper.h"

#ifdef SAIGA_USE_LIBTIFF

#    include "tiffio.h"
#    include "tiffio.hxx"

#    include <cstring>
#    include <iostream>
#    include <limits>
#    include <sstream>
#    include <vector>

namespace Saiga
{
namespace
{
struct ScopedTiff
{
    TIFF* tif;
    explicit ScopedTiff(TIFF* tif) : tif(tif) {}
    ~ScopedTiff()
    {
        if (tif) TIFFClose(tif);
    }
};

// Reads the first channel of the first directory of an open tiff into img.
bool LoadTiff(TIFF* tif, Image& img, const std::string& name)
{
    uint32_t width = 0, height = 0;
    uint16_t bitspersample = 0, sample_format = SAMPLEFORMAT_UINT, samples = 1, planar = PLANARCONFIG_CONTIG;

    // Get the image width and height
    if (!TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width))
    {
        std::cerr << "The file '" << name << "' has no 'TIFFTAG_IMAGEWIDTH'. The file is probably corrupted\n";
        return false;
    }
    if (!TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height))
    {
        std::cerr << "The file '" << name << "' has no 'TIFFTAG_IMAGELENGTH'. The file is probably corrupted\n";
        return false;
    }
    if (!TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitspersample))
    {
        std::cerr << "The file '" << name << "' has no 'TIFFTAG_BITSPERSAMPLE'. The file is probably corrupted\n";
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
    TIFFGetFieldDefaulted(tif, TIFFTAG_PLANARCONFIG, &planar);

    ImageType type = TYPE_UNKNOWN;
    if (sample_format == SAMPLEFORMAT_IEEEFP)
    {
        type = (bitspersample == 32) ? ImageType::F1 : (bitspersample == 64) ? ImageType::D1 : TYPE_UNKNOWN;
    }
    else if (sample_format == SAMPLEFORMAT_UINT)
    {
        type = (bitspersample == 8)    ? ImageType::UC1
               : (bitspersample == 16) ? ImageType::US1
               : (bitspersample == 32) ? ImageType::UI1
                                       : TYPE_UNKNOWN;
    }
    else if (sample_format == SAMPLEFORMAT_INT)
    {
        type = (bitspersample == 8)    ? ImageType::C1
               : (bitspersample == 16) ? ImageType::S1
               : (bitspersample == 32) ? ImageType::I1
                                       : TYPE_UNKNOWN;
    }

    if (type == TYPE_UNKNOWN)
    {
        std::cerr << "The file '" << name << "' has an unsupported pixel format (" << bitspersample
                  << " bits per sample, sample format " << sample_format << ")\n";
        return false;
    }
    if (samples == 0)
    {
        std::cerr << "The file '" << name << "' has 0 samples per pixel\n";
        return false;
    }
    if (width == 0 || height == 0 || width > uint32_t(std::numeric_limits<int>::max() / (8 * samples)) ||
        height > uint32_t(std::numeric_limits<int>::max()))
    {
        std::cerr << "The file '" << name << "' has an invalid size " << width << "x" << height << "\n";
        return false;
    }

    if (samples != 1)
    {
        std::cout << "Image '" << name << "' has more than one channel. Only loading first channel\n";
    }

    img.create(height, width, type);

    // TIFFReadScanline always writes a complete scanline, which contains all samples of a pixel for contiguous
    // data. Read into a separate buffer if that is larger than one row of the (single channel) output image.
    const size_t row_bytes       = size_t(width) * elementSize(type);
    const tmsize_t scanline_size = TIFFScanlineSize(tif);
    if (scanline_size <= 0)
    {
        return false;
    }
    const bool interleaved = samples > 1 && planar == PLANARCONFIG_CONTIG;
    if (!interleaved && size_t(scanline_size) != row_bytes)
    {
        std::cerr << "The file '" << name << "' has an unexpected scanline size\n";
        return false;
    }
    std::vector<uint8_t> scanline(interleaved ? size_t(scanline_size) : 0);
    if (interleaved && size_t(scanline_size) < row_bytes * samples)
    {
        std::cerr << "The file '" << name << "' has an unexpected scanline size\n";
        return false;
    }

    const size_t pixel_bytes = elementSize(type);
    for (uint32_t row = 0; row < height; ++row)
    {
        if (interleaved)
        {
            if (TIFFReadScanline(tif, scanline.data(), row, 0) != 1) return false;
            auto dst = (uint8_t*)img.rowPtr(row);
            for (uint32_t x = 0; x < width; ++x)
            {
                memcpy(dst + x * pixel_bytes, scanline.data() + size_t(x) * samples * pixel_bytes, pixel_bytes);
            }
        }
        else
        {
            // For separate planes, sample 0 is the first channel.
            if (TIFFReadScanline(tif, img.rowPtr(row), row, 0) != 1) return false;
        }
    }

    float xres = 0, yres = 0;
    bool has_xres = TIFFGetField(tif, TIFFTAG_XRESOLUTION, &xres);
    bool has_yres = TIFFGetField(tif, TIFFTAG_YRESOLUTION, &yres);

    if (has_xres && has_yres && xres > 0 && yres > 0)
    {
        uint16_t unit = RESUNIT_NONE;
        TIFFGetFieldDefaulted(tif, TIFFTAG_RESOLUTIONUNIT, &unit);

        vec2 pixel_size_mm(0, 0);

        if (unit == RESUNIT_CENTIMETER)
        {
            // xres is pixels/cm -> 10 / res = mm/pixel
            pixel_size_mm.x() = 10.f / xres;
            pixel_size_mm.y() = 10.f / yres;
        }
        else if (unit == RESUNIT_INCH)
        {
            // xres is pixels/inch -> 25.4 / res = mm/pixel
            pixel_size_mm.x() = 25.4f / xres;
            pixel_size_mm.y() = 25.4f / yres;
        }

        if (pixel_size_mm.x() > 0 && pixel_size_mm.y() > 0)
        {
            img.set_resolution(pixel_size_mm);
        }
    }
    return true;
}
}  // namespace

bool loadImageLibTiff(const std::filesystem::path& path, Image& img)
{
    TIFFSetWarningHandler(nullptr);
    TIFFSetWarningHandlerExt(nullptr);

#    ifdef _WIN32
    ScopedTiff tif(TIFFOpenW(path.c_str(), "r"));
#    else
    ScopedTiff tif(TIFFOpen(path.c_str(), "r"));
#    endif

    if (!tif.tif) return false;
    if (!LoadTiff(tif.tif, img, path.u8string()))
    {
        img.clear();
        return false;
    }
    return true;
}

bool saveImageLibTiff(const std::filesystem::path& path, const Image& img)
{
    TIFFSetWarningHandler(nullptr);
    TIFFSetWarningHandlerExt(nullptr);

#    ifdef _WIN32
    ScopedTiff scoped_tif(TIFFOpenW(path.c_str(), "w"));
#    else
    ScopedTiff scoped_tif(TIFFOpen(path.c_str(), "w"));
#    endif
    TIFF* tif = scoped_tif.tif;
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

    if (channels(img.type) == 1)
    {
        if (!TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK))
        {
            return false;
        }
    }
    else
    {
        if (!TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB))
        {
            return false;
        }
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
        case ImageType::UC3:
            sample_format = SAMPLEFORMAT_UINT;
            bitspersample = 8;
            samples       = 3;
            break;
        case ImageType::UC4:
            sample_format = SAMPLEFORMAT_UINT;
            bitspersample = 8;
            samples       = 4;
            break;
        case ImageType::US3:
            sample_format = SAMPLEFORMAT_UINT;
            bitspersample = 16;
            samples       = 3;
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

    auto resolution_mm = img.get_resolution();
    if (resolution_mm.has_value())
    {
        TIFFSetField(tif, TIFFTAG_RESOLUTIONUNIT, RESUNIT_CENTIMETER);

        // Pixels per centimeter
        TIFFSetField(tif, TIFFTAG_XRESOLUTION, 10.f / resolution_mm.value().x());
        TIFFSetField(tif, TIFFTAG_YRESOLUTION, 10.f / resolution_mm.value().y());
    }


    for (uint32_t row = 0; row < height; ++row)
    {
        if (TIFFWriteScanline(tif, const_cast<void*>(img.rowPtr(row)), row) != 1)
        {
            return false;
        }
    }

    // Writes the directory, so a full disk is reported here.
    return TIFFFlush(tif) == 1;
}

std::vector<uint8_t> saveImageLibTiff2Memory(const Image& img)
{
    TIFFSetWarningHandler(nullptr);
    TIFFSetWarningHandlerExt(nullptr);

    // Create an output string stream in binary mode
    std::ostringstream os(std::ios_base::binary);

    // Open the TIFF writing context using the C++ stream wrapper
    TIFF* tif = TIFFStreamOpen("MemoryTIFF", &os);
    if (!tif) return {};

    uint32_t width  = img.width;
    uint32_t height = img.height;

    // Get the image width and height
    if (!TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width))
    {
        TIFFClose(tif);
        return {};
    }
    if (!TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height))
    {
        TIFFClose(tif);
        return {};
    }

    if (channels(img.type) == 1)
    {
        if (!TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK))
        {
            TIFFClose(tif);
            return {};
        }
    }
    else
    {
        if (!TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB))
        {
            TIFFClose(tif);
            return {};
        }
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
        case ImageType::UC3:
            sample_format = SAMPLEFORMAT_UINT;
            bitspersample = 8;
            samples       = 3;
            break;
        case ImageType::UC4:
            sample_format = SAMPLEFORMAT_UINT;
            bitspersample = 8;
            samples       = 4;
            break;
        case ImageType::US3:
            sample_format = SAMPLEFORMAT_UINT;
            bitspersample = 16;
            samples       = 3;
            break;
        case ImageType::F1:
            sample_format = SAMPLEFORMAT_IEEEFP;
            bitspersample = 32;
            break;
        default:
            TIFFClose(tif);
            return {};
    }

    if (!TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, bitspersample))
    {
        TIFFClose(tif);
        return {};
    }

    if (!TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, sample_format))
    {
        TIFFClose(tif);
        return {};
    }

    if (!TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, samples))
    {
        TIFFClose(tif);
        return {};
    }

    auto resolution_mm = img.get_resolution();
    if (resolution_mm.has_value())
    {
        TIFFSetField(tif, TIFFTAG_RESOLUTIONUNIT, RESUNIT_CENTIMETER);

        // Pixels per centimeter
        TIFFSetField(tif, TIFFTAG_XRESOLUTION, 10.f / resolution_mm.value().x());
        TIFFSetField(tif, TIFFTAG_YRESOLUTION, 10.f / resolution_mm.value().y());
    }

    for (uint32_t row = 0; row < height; ++row)
    {
        if (TIFFWriteScanline(tif, const_cast<void*>(img.rowPtr(row)), row) != 1)
        {
            TIFFClose(tif);
            return {};
        }
    }

    // TIFFClose flushes the internal buffers and writes the final directory structure to the stream
    TIFFClose(tif);

    // Extract the binary string from the stream and copy it into the output vector
    std::string str_data = os.str();
    return std::vector<uint8_t>(str_data.begin(), str_data.end());
}

struct membuf : std::streambuf
{
    membuf(char const* base, size_t size)
    {
        char* p(const_cast<char*>(base));
        this->setg(p, p, p + size);
    }

    pos_type seekoff(off_type off, std::ios_base::seekdir dir,
                     std::ios_base::openmode which = std::ios_base::in) override
    {
        // Offsets come from the (untrusted) file. Positions outside of the buffer must fail, otherwise the next read
        // would access memory outside of it.
        off_type base = 0;
        if (dir == std::ios_base::cur)
            base = gptr() - eback();
        else if (dir == std::ios_base::end)
            base = egptr() - eback();
        off_type size = egptr() - eback();
        if (off < -base || off > size - base)
        {
            return pos_type(off_type(-1));
        }
        setg(eback(), eback() + base + off, egptr());
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
    ScopedTiff tif(TIFFStreamOpen("Tiff Loader", &strm));

    if (!tif.tif) return false;
    if (!LoadTiff(tif.tif, img, "memory"))
    {
        img.clear();
        return false;
    }
    return true;
}
}  // namespace Saiga

#endif