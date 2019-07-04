/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "freeimage.h"

#include "saiga/core/util/assert.h"

#ifdef SAIGA_USE_FREEIMAGE
#    include "internal/noGraphicsAPI.h"

#    include <FreeImagePlus.h>
#    include <cstring>

namespace Saiga
{
namespace FIP
{
bool load(const std::string& path, Image& img, ImageMetadata* metaData)
{
    fipImage fimg;
    if (!loadFIP(path, fimg))
    {
        return false;
    }

    if (metaData)
    {
        getMetaData(fimg, *metaData);
        //        printAllMetaData(fimg);
    }

    convert(fimg, img);

    return true;
}

bool loadFromMemory(ArrayView<const char> data, Image& img)
{
    fipImage fimg;

    fipMemoryIO fipmem((BYTE*)data.data(), data.size());
    if (!fimg.loadFromMemory(fipmem)) return false;
    convert(fimg, img);
    return true;
}


bool save(const std::string& path, const Image& img)
{
    fipImage fimg;
    convert(img, fimg);

    return saveFIP(path, fimg);
}

bool loadFIP(const std::string& path, fipImage& img)
{
    auto ret = img.load(path.c_str(), JPEG_EXIFROTATE);
    return ret;
}

bool saveFIP(const std::string& path, fipImage& img)
{
    // fipImage is not const to ensure compatibility with version 3.18.0 of freeimage
    auto ret = img.save(path.c_str());
    return ret;
}


void convert(const Image& _src, fipImage& dest)
{
    auto src = _src;

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


    //    int bpp = elementSize(src.type) * 8;
    int bpp = bitsPerPixel(src.type);


    FREE_IMAGE_TYPE t = FIT_UNKNOWN;

    switch (_src.type)
    {
        case UC1:
        case UC2:
        case UC3:
        case UC4:
            t = FIT_BITMAP;
            break;
        case US1:
            t = FIT_UINT16;
            break;
        case US3:
            t = FIT_RGB16;
            break;
        case F3:
            t = FIT_RGBF;
            break;
        default:
            break;
    }

    SAIGA_ASSERT(t != FIT_UNKNOWN);

    auto res = dest.setSize(t, src.width, src.height, bpp);

    SAIGA_ASSERT(res);


    // The FreeImage coordinate system is upside down relative to usual graphics conventions.
    // Thus, the scanlines are stored upside down, with the first scan in memory being the bottommost scan in the image.
    for (int i = 0; i < src.rows; ++i)
    {
        memcpy(dest.getScanLine(src.rows - i - 1), src.rowPtr(i), std::min<int>(dest.getScanWidth(), src.pitchBytes));
    }
}


void convert(const fipImage& src, Image& dest)
{
    SAIGA_ASSERT(src.isValid());
    dest.width  = src.getWidth();
    dest.height = src.getHeight();


    int channels = -1;
    switch (src.getColorType())
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



    if (src.getBitsPerPixel() == 32 && channels == 3)
    {
        channels = 4;
    }
    //    src.getImageType()

    int bitDepth = src.getBitsPerPixel() / channels;

    ImageElementType elementType = IET_ELEMENT_UNKNOWN;
    switch (src.getImageType())
    {
        case FIT_BITMAP:
            switch (bitDepth)
            {
                case 8:
                    elementType = IET_UCHAR;
                    break;
                case 16:
                    elementType = IET_USHORT;
                    break;
                case 32:
                    elementType = IET_UINT;
                    break;
            }
            break;
        case FIT_FLOAT:
        case FIT_RGBF:
        case FIT_RGBAF:
            elementType = IET_FLOAT;
            break;

        case FIT_RGB16:
        case FIT_RGBA16:
            elementType = IET_USHORT;
            break;
        default:
            break;
    }

    if (elementType == IET_ELEMENT_UNKNOWN)
    {
        throw std::runtime_error("Unknown FIT type " + std::to_string(src.getImageType()));
    }



    //    std::cout << "Channels: " << format.getChannels() << " BitsPerPixel: " << src.getBitsPerPixel() << " Bitdepth:
    //    " << format.getBitDepth() << std::endl;

    //    std::cout << format << std::endl;

    dest.type = getType(channels, elementType);
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


static double parseFraction(const void* data)
{
    const int* idata = reinterpret_cast<const int*>(data);

    return double(idata[0]) / double(idata[1]);
}

void getMetaData(fipImage& img, ImageMetadata& metaData)
{
    metaData.width  = img.getWidth();
    metaData.height = img.getHeight();

    fipTag tag;
    fipMetadataFind finder;
    if (finder.findFirstMetadata(FIMD_EXIF_MAIN, img, tag))
    {
        do
        {
            std::string t = tag.getKey();

            if (t == "DateTime")
            {
                metaData.DateTime = tag.toString(FIMD_EXIF_MAIN);
            }
            else if (t == "Make")
            {
                metaData.Make = (char*)tag.getValue();
            }
            else if (t == "Model")
            {
                metaData.Model = (char*)tag.getValue();
            }
            else
            {
                //                std::cout << "Tag: " << tag.getKey() << " Value: " << tag.toString(FIMD_EXIF_MAIN) <<
                //                std::endl;
            }

        } while (finder.findNextMetadata(tag));
    }

    // the class can be called again with another metadata model
    if (finder.findFirstMetadata(FIMD_EXIF_EXIF, img, tag))
    {
        do
        {
            std::string t = tag.getKey();
            if (t == "FocalLength")
            {
                metaData.FocalLengthMM = parseFraction(tag.getValue());
            }
            else if (t == "FocalLengthIn35mmFilm")
            {
                metaData.FocalLengthMM35 = reinterpret_cast<const short*>(tag.getValue())[0];
            }
            else if (t == "FocalPlaneResolutionUnit")
            {
                metaData.FocalPlaneResolutionUnit =
                    (ImageMetadata::ResolutionUnit) reinterpret_cast<const short*>(tag.getValue())[0];
            }
            else if (t == "FocalPlaneXResolution")
            {
                metaData.FocalPlaneXResolution = parseFraction(tag.getValue());
            }
            else if (t == "FocalPlaneYResolution")
                metaData.FocalPlaneYResolution = parseFraction(tag.getValue());
            else if (t == "ExposureTime")
                metaData.ExposureTime = parseFraction(tag.getValue());
            else if (t == "ISOSpeedRatings")
                metaData.ISOSpeedRatings = reinterpret_cast<const short*>(tag.getValue())[0];
            else
            {
                //                std::cout << "Tag: " << tag.getKey() << " Value: " << tag.toString(FIMD_EXIF_EXIF) <<
                //                std::endl;
            }
        } while (finder.findNextMetadata(tag));
    }
}


void printAllMetaData(fipImage& img)
{
    for (int i = 0; i <= 11; ++i)
    {
        FREE_IMAGE_MDMODEL model = (FREE_IMAGE_MDMODEL)i;
        std::cout << "Model: " << model << std::endl;
        fipTag tag;
        fipMetadataFind finder;
        if (finder.findFirstMetadata(model, img, tag))
        {
            do
            {
                std::string t = tag.getKey();

                std::cout << tag.getKey() << " : " << tag.toString(model) << " Type: " << tag.getType() << std::endl;


            } while (finder.findNextMetadata(tag));
        }
    }
}



}  // namespace FIP
}  // namespace Saiga


#endif
