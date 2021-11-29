/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "freeimage.h"

#include "saiga/core/util/assert.h"

#ifdef SAIGA_USE_FREEIMAGE
#    include "internal/noGraphicsAPI.h"

// #    include <FreeImagePlus.h>
#    include <FreeImage.h>
#    include <cstring>

namespace Saiga
{
static FREE_IMAGE_TYPE FIType(ImageType type)
{
    FREE_IMAGE_TYPE t = FIT_UNKNOWN;

    switch (type)
    {
        case UC1:
        case UC2:
        case UC3:
        case UC4:
            t = FIT_BITMAP;
            break;
            // === Short ===
        case US1:
            t = FIT_UINT16;
            break;
        case US3:
            t = FIT_RGB16;
            break;
        case US4:
            t = FIT_RGBA16;
            break;
            // === Float ===
        case F1:
            t = FIT_FLOAT;
            break;
        case F3:
            t = FIT_RGBF;
            break;
        case F4:
            t = FIT_RGBAF;
            break;
            // === Int ===
        case UI1:
            t = FIT_UINT32;
            break;
        case I1:
            t = FIT_INT32;
            break;
        default:
            break;
    }

    SAIGA_ASSERT(t != FIT_UNKNOWN);
    return t;
}

static int Channels(FREE_IMAGE_COLOR_TYPE type)
{
    int channels = -1;
    switch (type)
    {
        case FIC_MINISBLACK:
            channels = 1;
            break;
        case FIC_RGB:
            channels = 3;
            break;
        case FIC_RGBALPHA:
            channels = 4;
            break;
        default:
            break;
    }
    SAIGA_ASSERT(channels != -1);
    return channels;
}

#    if 0
void convert(const fipImage& src, Image& dest)
{
    SAIGA_ASSERT(src.isValid());
    dest.width  = src.getWidth();
    dest.height = src.getHeight();



    //    std::cout << "Channels: " << format.getChannels() << " BitsPerPixel: " << src.getBitsPerPixel() << " Bitdepth:
    //    " << format.getBitDepth() << std::endl;

    //    std::cout << format << std::endl;


    dest.create();

    // The FreeImage coordinate system is upside down relative to usual graphics conventions.
    // Thus, the scanlines are stored upside down, with the first scan in memory being the bottommost scan in the image.
    for (int i = 0; i < dest.rows; ++i)
    {
        memcpy(dest.rowPtr(dest.rows - i - 1), src.getScanLine(i), std::min<int>(dest.pitchBytes, src.getScanWidth()));
    }



#    if FREEIMAGE_COLORORDER == FREEIMAGE_COLORORDER_BGR
    if (dest.type == UC3)
    {
        ImageView<ucvec3> img = dest.getImageView<ucvec3>();
        img.swapChannels(0, 2);
    }
    if (dest.type == UC4)
    {
        ImageView<ucvec4> img = dest.getImageView<ucvec4>();
        img.swapChannels(0, 2);
    }
#    endif
}


bool loadFromMemory(ArrayView<const char> data, Image& img)
{
    fipImage fimg;

    fipMemoryIO fipmem((BYTE*)data.data(), data.size());
    if (!fimg.loadFromMemory(fipmem)) return false;
    convert(fimg, img);
    return true;
}

std::vector<unsigned char> saveToMemory(const Image& img, std::string file_extension)
{
    fipImage fimg;
    convert(img, fimg);

    FREE_IMAGE_FORMAT format = FreeImage_GetFIFFromFilename(file_extension.c_str());

    fipMemoryIO io;
    fimg.saveToMemory(format, io);

    unsigned char* data;
    unsigned int size;
    io.acquire(&data, &size);

    std::vector<unsigned char> result(size);
    memcpy(result.data(), data, size);
    return result;
}


#    endif
std::optional<Image> ImageIOLibFreeimage::LoadFromFile(const std::string& path, ImageLoadFlags flags)
{
    FREE_IMAGE_FORMAT format = FreeImage_GetFIFFromFilename(path.c_str());
    FIBITMAP* bm             = FreeImage_Load(format, path.c_str(), JPEG_EXIFROTATE | JPEG_ACCURATE);
    if (bm == nullptr)
    {
        return {};
    }


    FREE_IMAGE_COLOR_TYPE ftype = FreeImage_GetColorType(bm);
    FREE_IMAGE_TYPE itype       = FreeImage_GetImageType(bm);
    int channels                = Channels(ftype);
    int bpp                     = FreeImage_GetBPP(bm);
    if (bpp == 32 && channels == 3)
    {
        channels = 4;
    }
    int bitDepth = bpp / channels;
    int h        = FreeImage_GetHeight(bm);
    int w        = FreeImage_GetWidth(bm);

    ImageElementType elementType = ImageElementType::IET_ELEMENT_UNKNOWN;
    switch (itype)
    {
        case FIT_BITMAP:
            switch (bitDepth)
            {
                case 8:
                    elementType = ImageElementType::IET_UCHAR;
                    break;
                case 16:
                    elementType = ImageElementType::IET_USHORT;
                    break;
                case 32:
                    elementType = ImageElementType::IET_UINT;
                    break;
            }
            break;
        case FIT_FLOAT:
        case FIT_RGBF:
        case FIT_RGBAF:
            elementType = ImageElementType::IET_FLOAT;
            break;

        case FIT_UINT16:
        case FIT_RGB16:
        case FIT_RGBA16:
            elementType = ImageElementType::IET_USHORT;
            break;
        case FIT_INT32:
            elementType = ImageElementType::IET_INT;
            break;
        case FIT_UINT32:
            elementType = ImageElementType::IET_UINT;
            break;
        default:
            break;
    }

    if (elementType == ImageElementType::IET_ELEMENT_UNKNOWN)
    {
        FreeImage_Unload(bm);
        throw std::runtime_error("Unknown FIT type " + std::to_string(itype));
    }

    auto saiga_type = getType(channels, elementType);

    Image img(h, w, saiga_type);

    for (int i = 0; i < img.rows; ++i)
    {
        auto str_line    = FreeImage_GetScanLine(bm, i);
        size_t copy_size = w * (bpp / 8);
        memcpy(img.rowPtr(img.rows - i - 1), str_line, copy_size);
    }



#    if FREEIMAGE_COLORORDER == FREEIMAGE_COLORORDER_BGR
    if (img.type == UC3)
    {
        ImageView<ucvec3> v = img.getImageView<ucvec3>();
        v.swapChannels(0, 2);
    }
    if (img.type == UC4)
    {
        ImageView<ucvec4> v = img.getImageView<ucvec4>();
        v.swapChannels(0, 2);
    }
#    endif
    FreeImage_Unload(bm);
    return img;
}


bool ImageIOLibFreeimage::Save2File(const std::string& path, const Image& img, ImageSaveFlags flags)
{
    Image src = img;
#    if FREEIMAGE_COLORORDER == FREEIMAGE_COLORORDER_BGR
    if (src.type == UC3)
    {
        ImageView<ucvec3> img = src.getImageView<ucvec3>();
        img.swapChannels(0, 2);
    }
    if (src.type == UC4)
    {
        ImageView<ucvec4> img = src.getImageView<ucvec4>();
        img.swapChannels(0, 2);
    }
#    endif

    // fipImage is not const to ensure compatibility with version 3.18.0 of freeimage
    FREE_IMAGE_FORMAT format = FreeImage_GetFIFFromFilename(path.c_str());
    int bpp                  = bitsPerPixel(src.type);
    FREE_IMAGE_TYPE type     = FIType(src.type);
    SAIGA_ASSERT(format != FIF_UNKNOWN);

    FIBITMAP* bm = FreeImage_AllocateT(type, img.w, img.h, bpp);
    SAIGA_ASSERT(bm);

    // The FreeImage coordinate system is upside down relative to usual graphics conventions.
    // Thus, the scanlines are stored upside down, with the first scan in memory being the bottommost scan in the image.
    for (int i = 0; i < src.rows; ++i)
    {
        auto line = FreeImage_GetScanLine(bm, src.rows - i - 1);
        SAIGA_ASSERT(line);

        size_t copy_size = src.w * (bpp / 8);
        memcpy(line, src.rowPtr(i), copy_size);
    }


    bool ret = FreeImage_Save(format, bm, path.c_str());
    FreeImage_Unload(bm);
    return ret;
}


}  // namespace Saiga


#endif
