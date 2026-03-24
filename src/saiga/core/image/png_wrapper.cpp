/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "png_wrapper.h"

#include "saiga/core/util/assert.h"

#include <cstring>
#include <iostream>

#ifdef SAIGA_USE_PNG
#    include "internal/noGraphicsAPI.h"

#    include "png_types.h"
namespace Saiga
{

class PngException : public std::runtime_error
{
   public:
    PngException(const std::string& msg) : std::runtime_error(msg) {}
};
static void CxxPngErrorHandler(png_structp png_ptr, png_const_charp msg)
{
    // Throwing an exception satisfies libpng's requirement that this function never returns.
    throw PngException(msg ? msg : "Unknown libpng error");
}
static void CxxPngWarningHandler(png_structp png_ptr, png_const_charp msg)
{
    std::cerr << "libpng warning: " << (msg ? msg : "Unknown") << std::endl;
}

// RAII for reading
struct ScopedPngReader
{
    png_structp png = nullptr;
    png_infop info  = nullptr;

    ScopedPngReader()
    {
        // Notice we pass our custom error handlers here!
        png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, CxxPngErrorHandler, CxxPngWarningHandler);
        if (png)
        {
            info = png_create_info_struct(png);
        }
    }

    ~ScopedPngReader()
    {
        if (png)
        {
            png_destroy_read_struct(&png, info ? &info : nullptr, nullptr);
        }
    }

    bool is_valid() const { return png != nullptr && info != nullptr; }
};

// RAII for writing
struct ScopedPngWriter
{
    png_structp png = nullptr;
    png_infop info  = nullptr;

    ScopedPngWriter()
    {
        // Notice we pass our custom error handlers here!
        png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, CxxPngErrorHandler, CxxPngWarningHandler);
        if (png)
        {
            info = png_create_info_struct(png);
        }
    }

    ~ScopedPngWriter()
    {
        if (png)
        {
            png_destroy_write_struct(&png, info ? &info : nullptr);
        }
    }

    bool is_valid() const { return png != nullptr && info != nullptr; }
};

// RAII for C-style FILE pointers
struct ScopedFile
{
    FILE* fp = nullptr;
    ScopedFile(const std::filesystem::path& path, const char* mode)
    {
#    ifdef WIN32
        // Minor fix: _wfopen expects a wide string mode too
        std::wstring wmode(mode, mode + strlen(mode));
        fp = _wfopen(path.c_str(), wmode.c_str());
#    else
        fp = fopen(path.c_str(), mode);
#    endif
    }
    ~ScopedFile()
    {
        if (fp) fclose(fp);
    }
    operator FILE*() const { return fp; }
    bool is_valid() const { return fp != nullptr; }
};



static void setCompression(png_structp png_ptr, ImageCompression compression)
{
    switch (compression)
    {
        case ImageCompression::fast:
            png_set_compression_strategy(png_ptr, Z_RLE);
            png_set_filter(png_ptr, PNG_FILTER_TYPE_BASE, PNG_FILTER_NONE);
            png_set_compression_level(png_ptr, Z_BEST_SPEED);
            break;
        case ImageCompression::medium:
            png_set_compression_strategy(png_ptr, Z_RLE);
            png_set_filter(png_ptr, PNG_FILTER_TYPE_BASE, PNG_FILTER_SUB);
            png_set_compression_level(png_ptr, Z_DEFAULT_COMPRESSION);
            break;
        case ImageCompression::best:
            png_set_compression_strategy(png_ptr, Z_DEFAULT_STRATEGY);
            png_set_filter(png_ptr, PNG_FILTER_TYPE_BASE, PNG_FILTER_SUB);
            png_set_compression_level(png_ptr, Z_BEST_COMPRESSION);
            break;
        default:
            SAIGA_EXIT_ERROR("Unknown Compression");
    }
}

bool ImageIOLibPNG::Save2File(const std::filesystem::path& path, const Image& img, ImageSaveFlags flags)
{
    ScopedFile file(path, "wb");
    if (!file.is_valid())
    {
        std::cerr << "could not open file: " << path.string() << std::endl;
        return false;
    }

    ScopedPngWriter writer;
    if (!writer.is_valid())
    {
        std::cerr << "could not allocate libpng write structs" << std::endl;
        return false;
    }

    try
    {
        // If ANY libpng function fails below this line, it throws PngException.
        // The catch block will catch it, and ScopedFile/ScopedPngWriter destructors
        // will safely release the memory and close the file.

        png_init_io(writer.png, file);

        auto resolution_mm = img.get_resolution();
        if (resolution_mm.has_value())
        {
            png_uint_32 xres = (png_uint_32)(1000.f / resolution_mm.value().x() + 0.5f);
            png_uint_32 yres = (png_uint_32)(1000.f / resolution_mm.value().y() + 0.5f);
            png_set_pHYs(writer.png, writer.info, xres, yres, PNG_RESOLUTION_METER);
        }

        int bit_depth  = bitsPerChannel(img.type);
        int color_type = PngColorType(img.type);

        png_set_IHDR(writer.png, writer.info, img.width, img.height, bit_depth, color_type, PNG_INTERLACE_NONE,
                     PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

        setCompression(writer.png, flags.compression);

        png_write_info(writer.png, writer.info);

        png_set_packing(writer.png);
        png_set_swap(writer.png);

        bool invertY = false;

        for (int i = 0; i < img.height; i++)
        {
            auto rowPtr = (png_byte*)img.rowPtr(invertY ? img.height - i - 1 : i);
            png_write_row(writer.png, rowPtr);
        }

        png_write_end(writer.png, NULL);

        return true;
    }
    catch (const PngException& e)
    {
        std::cerr << "Failed to save PNG: " << e.what() << std::endl;
        return false;
    }
}
struct TPngDestructor
{
    png_struct* p;
    png_infop info_ptr = nullptr;
    TPngDestructor(png_struct* p) : p(p) {}
    ~TPngDestructor()
    {
        if (p)
        {
            png_destroy_write_struct(&p, &info_ptr);
        }
    }
};

static void PngWriteCallback(png_structp png_ptr, png_bytep data, png_size_t length)
{
    std::vector<unsigned char>* p = (std::vector<unsigned char>*)png_get_io_ptr(png_ptr);
    p->insert(p->end(), data, data + length);
}


std::vector<unsigned char> ImageIOLibPNG::Save2Memory(const Image& img, ImageSaveFlags flags)
{
    ScopedPngWriter writer;
    if (!writer.is_valid())
    {
        std::cerr << "could not allocate libpng write structs" << std::endl;
        return {};
    }

    try
    {
        int bit_depth  = bitsPerChannel(img.type);
        int color_type = PngColorType(img.type);

        png_set_IHDR(writer.png, writer.info, img.width, img.height, bit_depth, color_type, PNG_INTERLACE_NONE,
                     PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

        setCompression(writer.png, flags.compression);

        std::vector<unsigned char*> rows(img.height);
        for (size_t y = 0; y < img.height; ++y) // Note: changed img.h to img.height to match your other code
        {
            rows[y] = (unsigned char*)img.rowPtr(y);
        }

        auto resolution_mm = img.get_resolution();
        if (resolution_mm.has_value())
        {
            png_uint_32 xres = (png_uint_32)(1000.f / resolution_mm.value().x() + 0.5f);
            png_uint_32 yres = (png_uint_32)(1000.f / resolution_mm.value().y() + 0.5f);
            png_set_pHYs(writer.png, writer.info, xres, yres, PNG_RESOLUTION_METER);
        }

        std::vector<unsigned char> out_data;
        png_set_rows(writer.png, writer.info, rows.data());
        png_set_write_fn(writer.png, &out_data, PngWriteCallback, NULL);
        png_write_png(writer.png, writer.info, PNG_TRANSFORM_IDENTITY, NULL);

        return out_data;
    }
    catch (const PngException& e)
    {
        std::cerr << "Failed to save PNG to memory: " << e.what() << std::endl;
        return {};
    }
}

std::optional<Image> ImageIOLibPNG::LoadFromFile(const std::filesystem::path& path, ImageLoadFlags flags)
{
    ScopedFile file(path, "rb");
    if (!file.is_valid()) return {};

    ScopedPngReader reader;
    if (!reader.is_valid()) return {};

    try
    {
        png_init_io(reader.png, file);
        png_set_sig_bytes(reader.png, 0);  // Assuming 0 sig bytes read
        png_read_info(reader.png, reader.info);

        png_uint_32 pw, ph;
        int bit_depth, color_type, interlace_type;
        png_get_IHDR(reader.png, reader.info, &pw, &ph, &bit_depth, &color_type, &interlace_type, NULL, NULL);

        SAIGA_ASSERT(interlace_type == PNG_INTERLACE_NONE);

        // ... color transformations ...
        if (color_type == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(reader.png);
        if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) png_set_expand_gray_1_2_4_to_8(reader.png);
        if (png_get_valid(reader.png, reader.info, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(reader.png);

        png_read_update_info(reader.png, reader.info);
        png_get_IHDR(reader.png, reader.info, &pw, &ph, &bit_depth, &color_type, &interlace_type, NULL, NULL);

        unsigned int row_bytes = png_get_rowbytes(reader.png, reader.info);
        int rowAlignment       = 4;
        int rowPadding         = (rowAlignment - (row_bytes % rowAlignment)) % rowAlignment;
        int bytesPerRow        = row_bytes + rowPadding;

        if (bit_depth > 8) png_set_swap(reader.png);
        png_set_packing(reader.png);

        Image img;
        img.create(ph, pw, bytesPerRow, saigaType(color_type, bit_depth));
        img.makeZero();

        for (int i = 0; i < img.height; i++)
        {
            auto rowPtr = (png_byte*)img.rowPtr(i);
            // If this fails, it throws PngException.
            // reader and file destructors clean everything up automatically.
            png_read_row(reader.png, rowPtr, nullptr);
        }

        png_uint_32 res_x, res_y;
        int unit_type;

        if (png_get_pHYs(reader.png, reader.info, &res_x, &res_y, &unit_type))
        {
            if (unit_type == PNG_RESOLUTION_METER && res_x > 0 && res_y > 0)
            {
                // Assuming vec2 is your math vector type
                vec2 pixelSizeMM = vec2(1000.f / res_x, 1000.f / res_y);
                img.set_resolution(pixelSizeMM);
            }
        }

        return img;
    }
    catch (const PngException& e)
    {
        std::cerr << "Failed to load PNG: " << e.what() << std::endl;
        return {};  // Return empty optional on failure
    }
}
struct PngMemoryStreamHelper
{
    const char* data;
    size_t pos = 0;
    size_t size;
};
static void ReadDataFromInputStream(png_structp png_ptr, png_bytep outBytes, png_size_t byteCountToRead)
{
    png_voidp io_ptr = png_get_io_ptr(png_ptr);
    if (io_ptr == NULL) return;

    PngMemoryStreamHelper& stream = *(PngMemoryStreamHelper*)io_ptr;

    // Check for bounds. If we over-read, trigger a libpng error (which throws PngException)
    if (stream.pos + byteCountToRead > stream.size)
    {
        png_error(png_ptr, "Unexpected end of memory stream while reading PNG.");
        return;
    }

    memcpy(outBytes, stream.data + stream.pos, byteCountToRead);
    stream.pos += byteCountToRead;
}

std::optional<Image> ImageIOLibPNG::LoadFromMemory(const void* data, size_t size, ImageLoadFlags flags)
{
    // 1. Automatically allocates png and info structs, binding our C++ exception handlers
    ScopedPngReader reader;
    if (!reader.is_valid())
    {
        std::cerr << "Failed to allocate libpng read structs." << std::endl;
        return {};
    }

    try
    {
        PngMemoryStreamHelper input_stream;
        input_stream.data = (char*)data;
        input_stream.size = size;
        png_set_read_fn(reader.png, &input_stream, ReadDataFromInputStream);

        unsigned int sig_read = 0;
        png_set_sig_bytes(reader.png, sig_read);

        png_read_info(reader.png, reader.info);

        png_uint_32 pw, ph;
        int bit_depth;
        int color_type;
        int interlace_type;
        png_get_IHDR(reader.png, reader.info, &pw, &ph, &bit_depth, &color_type, &interlace_type, NULL, NULL);
        SAIGA_ASSERT(interlace_type == PNG_INTERLACE_NONE);

        if (color_type == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(reader.png);

        if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) png_set_expand_gray_1_2_4_to_8(reader.png);

        if (png_get_valid(reader.png, reader.info, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(reader.png);

        // update the color_type data because it might have changed due to the calls above
        png_read_update_info(reader.png, reader.info);
        png_get_IHDR(reader.png, reader.info, &pw, &ph, &bit_depth, &color_type, &interlace_type, NULL, NULL);


        unsigned int row_bytes = png_get_rowbytes(reader.png, reader.info);

        // we want to row-align the image in our output data
        int rowAlignment = 4;
        int rowPadding   = (rowAlignment - (row_bytes % rowAlignment)) % rowAlignment;
        int bytesPerRow  = row_bytes + rowPadding;

        if (bit_depth > 8)
        {
            png_set_swap(reader.png);
        }

        png_set_packing(reader.png);

        Image img;
        img.create(ph, pw, bytesPerRow, saigaType(color_type, bit_depth));
        img.makeZero();

        png_uint_32 res_x, res_y;
        int unit_type;

        if (png_get_pHYs(reader.png, reader.info, &res_x, &res_y, &unit_type))
        {
            if (unit_type == PNG_RESOLUTION_METER && res_x > 0 && res_y > 0)
            {
                vec2 pixelSizeMM = vec2(1000.f / res_x, 1000.f / res_y);
                img.set_resolution(pixelSizeMM);
            }
        }

        // --- READ IMAGE DATA ---

        for (int i = 0; i < img.height; i++)
        {
            auto rowPtr = (png_byte*)img.rowPtr(i);
            // If the memory stream runs out or gets corrupted, this throws PngException
            png_read_row(reader.png, rowPtr, nullptr);
        }

        // Return safely. ScopedPngReader's destructor runs automatically here
        // and safely destroys reader.png and reader.info.
        return img;
    }
    catch (const PngException& e)
    {
        std::cerr << "Failed to load PNG from memory: " << e.what() << std::endl;

        // Return empty optional. ScopedPngReader cleans up libpng structs automatically.
        return {};
    }
}

}  // namespace Saiga
#endif
