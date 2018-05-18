/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/image/freeimage.h"
#include "saiga/util/assert.h"

#ifdef SAIGA_USE_FREEIMAGE
#include <FreeImagePlus.h>
#include <cstring>

namespace Saiga {
namespace FIP {


bool load(const std::string &path, Image &img, ImageMetadata *metaData)
{
    fipImage fimg;
    if(!loadFIP(path,fimg)){
        return false;
    }

    if(metaData){
        getMetaData(fimg,*metaData);
        //        printAllMetaData(fimg);
    }

    convert(fimg,img);

    return true;
}

bool save(const std::string &path, const Image &img)
{
    fipImage fimg;
    convert(img,fimg);

    return saveFIP(path,fimg);
}

bool loadFIP(const std::string &path, fipImage &img){
    auto ret = img.load(path.c_str(),JPEG_EXIFROTATE);
    return ret;
}

bool saveFIP(const std::string &path, const fipImage &img){
    auto ret = img.save(path.c_str());
    return ret;
}


void convert(const Image &_src, fipImage &dest)
{

    auto src = _src;

#if FREEIMAGE_COLORORDER == FREEIMAGE_COLORORDER_BGR
    if(src.type == UC3)
    {
        ImageView<ucvec3> img = src.getImageView<ucvec3>();
        img.swapChannels(0,2);
    }
    if(src.type == UC4)
    {
        ImageView<ucvec4> img = src.getImageView<ucvec4>();
        img.swapChannels(0,2);
    }
#endif


    int bpp = elementSize(src.type) * 8;


    FREE_IMAGE_TYPE t = FIT_BITMAP;

    switch(_src.type)
    {
        case US1:
        t = FIT_UINT16;
        break;
    default:
        break;
    }

    auto res = dest.setSize(
                t,
                src.width,src.height,
                bpp
                );

    SAIGA_ASSERT(res);


    for(int i =0; i < src.rows; ++i)
    {
        memcpy(dest.getScanLine(i),src.rowPtr(i), std::min<int>(dest.getScanWidth(),src.pitchBytes));
    }

}


void convert(const fipImage &src, Image& dest){

    SAIGA_ASSERT(src.isValid());
    dest.width = src.getWidth();
    dest.height = src.getHeight();


    int channels = -1;
    switch(src.getColorType()){
    case FIC_MINISBLACK:
        channels = 1;
        break;
    case FIC_RGB:
        channels = 3;
        break;
    case FIC_RGBALPHA:
        channels = 4;
        break;
    }
    SAIGA_ASSERT(channels != -1);




    if(src.getBitsPerPixel() == 32 && channels ==3)
    {
        channels = 4;
    }


    int bitDepth= src.getBitsPerPixel() / channels;
    ImageElementType elementType = IET_ELEMENT_UNKNOWN;
    switch(bitDepth)
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
    SAIGA_ASSERT(elementType != IET_ELEMENT_UNKNOWN);


    //    cout << "Channels: " << format.getChannels() << " BitsPerPixel: " << src.getBitsPerPixel() << " Bitdepth: " << format.getBitDepth() << endl;

    //    cout << format << endl;

    dest.type =  getType(channels,elementType);
    dest.create();

    for(int i =0; i < dest.rows; ++i)
    {
        memcpy(dest.rowPtr(i),src.getScanLine(i), std::min<int>(dest.pitchBytes,src.getScanWidth()));
    }



#if FREEIMAGE_COLORORDER == FREEIMAGE_COLORORDER_BGR
    if(dest.type == UC3)
    {
        ImageView<ucvec3> img = dest.getImageView<ucvec3>();
        img.swapChannels(0,2);
    }
    if(dest.type == UC4)
    {
        ImageView<ucvec4> img = dest.getImageView<ucvec4>();
        img.swapChannels(0,2);
    }
#endif

}

#endif


static double parseFraction(const void* data){
    const int* idata = reinterpret_cast<const int*>(data);

    return double(idata[0]) / double(idata[1]);
}

void getMetaData(fipImage &img, ImageMetadata& metaData){
    metaData.width = img.getWidth();
    metaData.height = img.getHeight();

    fipTag tag;
    fipMetadataFind finder;
    if( finder.findFirstMetadata(FIMD_EXIF_MAIN, img, tag) ) {
        do {
            std::string t = tag.getKey();

            if(t == "DateTime"){
                metaData.DateTime = tag.toString(FIMD_EXIF_MAIN);
            }else if(t == "Make"){
                metaData.Make = (char*)tag.getValue();
            }else if(t == "Model"){
                metaData.Model = (char*)tag.getValue();
            }else{
                //                cout << "Tag: " << tag.getKey() << " Value: " << tag.toString(FIMD_EXIF_MAIN) << endl;
            }

        } while( finder.findNextMetadata(tag) );
    }

    // the class can be called again with another metadata model
    if( finder.findFirstMetadata(FIMD_EXIF_EXIF, img, tag) ) {
        do {
            std::string t = tag.getKey();
            if(t == "FocalLength"){
                metaData.FocalLengthMM = parseFraction(tag.getValue());
            }else if(t == "FocalLengthIn35mmFilm"){
                metaData.FocalLengthMM35 = reinterpret_cast<const short*>(tag.getValue())[0];
            }else if(t == "FocalPlaneResolutionUnit"){
                metaData.FocalPlaneResolutionUnit = (ImageMetadata::ResolutionUnit) reinterpret_cast<const short*>(tag.getValue())[0];
            }else if(t == "FocalPlaneXResolution"){
                metaData.FocalPlaneXResolution = parseFraction(tag.getValue());
            }else if(t == "FocalPlaneYResolution"){
                metaData.FocalPlaneYResolution = parseFraction(tag.getValue());
            }else{
                //                cout << "Tag: " << tag.getKey() << " Value: " << tag.toString(FIMD_EXIF_MAIN) << endl;
            }
        } while( finder.findNextMetadata(tag) );
    }




}


void printAllMetaData(fipImage &img)
{
    for(int i = -1; i <= 11; ++i){
        FREE_IMAGE_MDMODEL model = (FREE_IMAGE_MDMODEL)i;
        cout << "Model: " << model << endl;
        fipTag tag;
        fipMetadataFind finder;
        if( finder.findFirstMetadata(model, img, tag) ) {
            do {
                std::string t = tag.getKey();

                cout << tag.getKey() << " : " << tag.toString(model) << " Type: " << tag.getType() << endl;


            } while( finder.findNextMetadata(tag) );
        }
    }

}


}
}
