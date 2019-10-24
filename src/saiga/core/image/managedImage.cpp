/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/image/managedImage.h"

#include "saiga/core/util/assert.h"

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
        std::cout << "could not find " << _path << std::endl;
    }

    bool erg         = false;
    std::string type = fileEnding(path);

    if (type == "saigai")
    {
        // saiga raw image format
        return loadRaw(path);
    }

#ifdef SAIGA_USE_PNG
    // use libpng for png images
    if (type == "png")
    {
        return PNG::load(*this, path, false);
    }
#endif


    // use libfreeimage if available
#ifdef SAIGA_USE_FREEIMAGE
    erg = FIP::load(path, *this, 0);
    return erg;
#endif

    // as a last resort use stb_image.h from the internals directory
    erg = loadImageSTB(path, *this);
    return erg;
}

bool Image::loadFromMemory(ArrayView<const char> data)
{
    bool erg = false;

#ifdef SAIGA_USE_FREEIMAGE
    erg = FIP::loadFromMemory(data, *this);
    return erg;
#endif


    return erg;
}

bool Image::save(const std::string& path) const
{
    SAIGA_ASSERT(valid());

    std::string type = fileEnding(path);

    if (type == "saigai")
    {
        // saiga raw image format
        return saveRaw(path);
    }


#ifdef SAIGA_USE_PNG
    // use libpng for png images
    if (type == "png")
    {
        return PNG::save(*this, path, false);
    }
#endif

#ifdef SAIGA_USE_FREEIMAGE
    return FIP::save(path, *this);
#endif

    // as a last resort use stb_image.h from the internals directory
    return saveImageSTB(path, *this);
}

#define SAIGA_BINARY_IMAGE_MAGIC_NUMBER 8574385


bool Image::loadRaw(const std::string& path)
{
    clear();
    std::fstream stream;

    try
    {
        stream.open(path, std::ios::in | std::ios::binary);
    }
    catch (const std::fstream::failure& e)
    {
        std::cout << e.what() << std::endl;
        std::cout << "Exception opening/reading file\n";
        return false;
    }

    int magic;
    stream.read((char*)&magic, sizeof(int));
    stream.read((char*)&width, sizeof(int));
    stream.read((char*)&height, sizeof(int));
    stream.read((char*)&type, sizeof(int));
    pitchBytes = 0;

    SAIGA_ASSERT(magic == SAIGA_BINARY_IMAGE_MAGIC_NUMBER);
    SAIGA_ASSERT(type != TYPE_UNKNOWN);


    create();

    int es = elementSize(type);
    for (int i = 0; i < height; ++i)
    {
        // store it compact
        stream.read((char*)rowPtr(i), width * es);
    }

    stream.close();

    return true;
}

bool Image::saveRaw(const std::string& path) const
{
    std::fstream stream;

    try
    {
        stream.open(path, std::ios::out | std::ios::binary);
    }
    catch (const std::fstream::failure& e)
    {
        std::cout << e.what() << std::endl;
        std::cout << "Exception opening/reading file\n";
        return false;
    }

    int magic = 8574385;
    stream.write((char*)&magic, sizeof(int));
    stream.write((char*)&width, sizeof(int));
    stream.write((char*)&height, sizeof(int));
    stream.write((char*)&type, sizeof(int));

    int es = elementSize(type);
    for (int i = 0; i < height; ++i)
    {
        // store it compact
        stream.write((char*)rowPtr(i), width * es);
    }
    stream.flush();
    stream.close();

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
       << " channels/elementType " << channels(f.type) << "/" << elementType(f.type)
       << " BPP: " << bitsPerPixel(f.type);
    return os;
}

}  // namespace Saiga
