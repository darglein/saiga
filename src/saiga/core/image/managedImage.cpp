/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/image/managedImage.h"

#include "saiga/core/util/BinaryFile.h"
#include "saiga/core/util/assert.h"
#include "saiga/core/util/file.h"
#include "saiga/core/util/zlib.h"

// for the load and save function
#include "saiga/core/image/freeimage.h"
#include "saiga/core/image/png_wrapper.h"
#include "saiga/core/image/templatedImage.h"
#include "saiga/core/math/imath.h"
#include "saiga/core/util/color.h"
#include "saiga/core/util/fileChecker.h"
#include "saiga/core/util/tostring.h"

#include "internal/noGraphicsAPI.h"
#include "internal/stb_image_read_wrapper.h"
#include "internal/stb_image_write_wrapper.h"

#include <fstream>
namespace Saiga
{
Image::Image(int h, int w, ImageType type)
    : ImageBase(h, w, iAlignUp(elementSize(type) * w, DEFAULT_ALIGNMENT)), type(type)
{
    create();
}

Image::Image(ImageDimensions dimensions, ImageType type) : Image(dimensions.h, dimensions.w, type) {}


void Image::create()
{
    SAIGA_ASSERT(width > 0 && height > 0 && type != TYPE_UNKNOWN);

    if (pitchBytes == 0)
    {
        pitchBytes = iAlignUp(elementSize(type) * width, DEFAULT_ALIGNMENT);
    }

    vdata.resize(size());

    SAIGA_ASSERT(valid());
}

void Image::create(ImageDimensions dimensions)
{
    create(dimensions.h, dimensions.w);
}

void Image::create(int h, int w)
{
    pitchBytes = 0;
    height     = h;
    width      = w;
    create();
}

void Image::create(int h, int w, ImageType t)
{
    pitchBytes = 0;
    height     = h;
    width      = w;
    type       = t;
    create();
}

void Image::create(int h, int w, int p, ImageType t)
{
    pitchBytes = p;
    create(h, w, t);
}

void Image::clear()
{
    (*this) = Image();
}

void Image::free()
{
    pitchBytes = 0;
    vdata.clear();
    vdata.shrink_to_fit();
}

void Image::makeZero()
{
    std::fill(vdata.begin(), vdata.end(), 0);
}

bool Image::valid() const
{
    return width > 0 && height > 0 && pitchBytes > 0 && type != TYPE_UNKNOWN && size() == vdata.size();
}



bool Image::load(const std::string& _path)
{
    clear();

    auto path = SearchPathes::image(_path);

    if (path.empty())
    {
        //        std::cout << "could not find " << _path << std::endl;
        return false;
    }

    bool erg         = false;
    std::string type = fileEnding(path);


    if (type == "saigai")
    {
        // saiga raw image format
        return loadRaw(path);
    }

    // use libpng for png images
    if (type == "png")
    {
#ifdef SAIGA_USE_PNG
        ImageIOLibPNG io;
        auto result = io.LoadFromFile(path);
        if (result.has_value())
        {
            *this = result.value();
            return true;
        }
        else
        {
            return false;
        }

#else
        std::cerr << "Warning: Using .png without libpng. This might be slow." << std::endl;
#endif
    }


    // use libfreeimage if available
#ifdef SAIGA_USE_FREEIMAGE
    {
        ImageIOLibFreeimage io;
        auto result = io.LoadFromFile(path);
        if (result.has_value())
        {
            *this = result.value();
            return true;
        }
        else
        {
            return false;
        }
    }
#endif

    // as a last resort use stb_image.h from the internals directory
    erg = loadImageSTB(path, *this);
    return erg;
}

bool Image::loadFromMemory(ArrayView<const char> data, const std::string& hint)
{
    bool erg = false;

    if (hint == "png")
    {
#ifdef SAIGA_USE_PNG

#endif
    }
#ifdef SAIGA_USE_FREEIMAGE
    SAIGA_EXIT_ERROR("not implemented");
    // erg = FIP::loadFromMemory(data, *this);
    // return erg;
#endif


    return erg;
}

std::vector<unsigned char> Image::saveToMemory(std::string file_extension) const
{
    std::vector<unsigned char> result;

    if (file_extension == "png")
    {
#ifdef SAIGA_USE_PNG


        ImageIOLibPNG io;
        return io.Save2Memory(*this);
        SAIGA_EXIT_ERROR("png not implemneted");
#endif
    }
#ifdef SAIGA_USE_FREEIMAGE
    SAIGA_EXIT_ERROR("not implemented");
    // return Saiga::FIP::saveToMemory(*this, file_extension);
#endif

    return result;
}

bool Image::save(const std::string& path) const
{
    SAIGA_ASSERT(valid());

    std::string output_type = fileEnding(path);

    if (output_type == "saigai")
    {
        // saiga raw image format
        return saveRaw(path);
    }

    if (output_type == "jpg" && channels(this->type) != 3)
    {
        std::cerr << "jpg is only supported with 3 channels" << std::endl;
        return false;
    }

    if (output_type == "png")
    {
#ifdef SAIGA_USE_PNG
        // return LibPNG::save(path, *this, false);

        ImageIOLibPNG io;
        return io.Save2File(path, *this);

#else
        std::cerr << "Warning: Using .png without libpng. This might be slow." << std::endl;
#endif
    }

#ifdef SAIGA_USE_FREEIMAGE
    {
        ImageIOLibFreeimage io;
        return io.Save2File(path, *this);
    }
    // return FIP::save(path, *this);
#endif

    // as a last resort use stb_image.h from the internals directory
    return saveImageSTB(path, *this);
}

constexpr int saiga_image_magic_number            = 8574385;
constexpr int saiga_compressed_image_magic_number = 198760233;
constexpr size_t saiga_image_header_size          = 4 * sizeof(int);



bool Image::loadRaw(const std::string& path)
{
    clear();



    auto data = File::loadFileBinary(path);
    BinaryInputVector stream(data.data(), data.size());

    int magic;
    stream >> magic >> width >> height >> type;



    bool compress = false;
    if (magic == saiga_image_magic_number)
    {
        compress = false;
    }
    else if (magic == saiga_compressed_image_magic_number)
    {
        compress = true;
    }
    else
    {
        SAIGA_EXIT_ERROR("invalid magic number");
    }

    pitchBytes = 0;
    create();
    SAIGA_ASSERT(type != TYPE_UNKNOWN);
    int es = elementSize(type);

    if (compress)
    {
#ifdef SAIGA_USE_ZLIB
        auto uncompressed = Saiga::uncompress(stream.data + stream.current);
        size_t line_size  = width * es;
        for (int i = 0; i < height; ++i)
        {
            size_t offset = i * width * es;
            memcpy(rowPtr(i), uncompressed.data() + offset, line_size);
        }
#else
        SAIGA_EXIT_ERROR("zlib required!");
#endif
    }
    else
    {
        for (int i = 0; i < height; ++i)
        {
            stream.read((char*)rowPtr(i), width * es);
        }
    }

    return true;
}

bool Image::saveRaw(const std::string& path, bool do_compress) const
{
    BinaryOutputVector stream;

    int magic = do_compress ? saiga_compressed_image_magic_number : saiga_image_magic_number;
    stream << magic << width << height << type;
    SAIGA_ASSERT(stream.data.size() == saiga_image_header_size);

    int es = elementSize(type);
    for (int i = 0; i < height; ++i)
    {
        // store it compact
        stream.write((char*)rowPtr(i), width * es);
    }

#ifdef SAIGA_USE_ZLIB
    if (do_compress)
    {
        auto compressed_data =
            Saiga::compress(stream.data.data() + saiga_image_header_size, stream.data.size() - saiga_image_header_size);

        std::ofstream is(path, std::ios::binary | std::ios::out);
        is.write(stream.data.data(), saiga_image_header_size);
        is.write((const char*)compressed_data.data(), compressed_data.size());
    }
    else
#endif
    {
        File::saveFileBinary(path, stream.data.data(), stream.data.size());
    }
    return true;
}

bool Image::saveConvert(const std::string& path, float minValue, float maxValue)
{
    if (type == ImageType::F1)
    {
        Saiga::TemplatedImage<ucvec4> i(h, w);
        Saiga::ImageTransformation::depthToRGBA(getImageView<float>(), i.getImageView(), minValue, maxValue);
        i.save(path);
    }

    return false;
}

std::vector<uint8_t> Image::compress()
{
    return compressImageSTB(*this);
}

void Image::decompress(std::vector<uint8_t> data)
{
    decompressImageSTB(*this, data);
}

std::ostream& operator<<(std::ostream& os, const Image& f)
{
    os << "Image " << f.width << "x" << f.height << " "
       << " pitch " << f.pitchBytes << " "
       << " channels/elementType " << channels(f.type) << "/" << (int)elementType(f.type)
       << " BPP: " << bitsPerPixel(f.type);
    return os;
}
vec4 Image::texture(vec2 uv)
{
    switch (type)
    {
        case ImageType::UC3:
            return make_vec4(getImageView<ucvec3>().interUVGL(uv(0), uv(1)).cast<float>() * (1.f / 255.f), 1);
            break;
        case ImageType::UC4:
            return getImageView<ucvec4>().interUVGL(uv(0), uv(1)).cast<float>() * (1.f / 255.f);
            break;
        default:
            SAIGA_EXIT_ERROR("texture not implemented for this image type");
    }
    return Saiga::vec4(0, 0, 0, 0);
}

}  // namespace Saiga
