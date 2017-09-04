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

FREE_IMAGE_TYPE getFIT2(ImageFormat format){
    if(format.getBitDepth()==16 && format.getChannels()==3){
        return FIT_RGB16;
    }else if(format.getBitDepth()==16 && format.getChannels()==4){
        return FIT_RGBA16;
    }else if(format.getBitDepth()==16 && format.getChannels()==1){
        return FIT_UINT16;
    }else if(format.getBitDepth()==32 && format.getChannels()==1){
        return FIT_UINT32;
    }

    return FIT_BITMAP;
}


void convert(const Image &_src, fipImage &dest){
    auto src = _src;
    //    if(src.Format().getChannels() == 1 && src.Format().bitsPerPixel()==8){

    ////    }else if(src.Format().getChannels() == 3 && src.Format().bitsPerPixel()==24){
    //        }else if(src.Format().getChannels() == 3 ){
    //        src.flipRB();
    //    }else if(src.Format().getChannels() == 4){
    //        src.flipRB();
    //    }else{
    //        std::cout<<"INVALID FORMAT: channels: " << src.Format().getChannels() << ", bitsperpixel " << src.Format().bitsPerPixel() <<std::endl;
    //        SAIGA_ASSERT(0);
    //    }

#if FREEIMAGE_COLORORDER == FREEIMAGE_COLORORDER_BGR
    //convert RGB -> BGR
    if(src.Format().getChannels() == 3 || src.Format().getChannels() == 4){
        src.flipRB();
    }
#endif


    dest.setSize(getFIT2(src.Format()),src.width,src.height,src.Format().bitsPerPixel());

    auto data = dest.accessPixels();


    //    memcpy(data,src.getRawData(),src.getSize());

    for(int y = 0; y < src.height; ++y){

        auto srcPtr = src.positionPtr(0,y);
        auto targetPtr = data + y * dest.getScanWidth();
        memcpy(targetPtr,srcPtr,dest.getScanWidth());
    }

}


void convert(const fipImage &src, Image& dest){
    SAIGA_ASSERT(src.isValid());
    dest.width = src.getWidth();
    dest.height = src.getHeight();

    ImageFormat format;

    switch(src.getColorType()){
    case FIC_MINISBLACK:
        format.setChannels(1);
        break;
    case FIC_RGB:
        format.setChannels(3);
        break;
    case FIC_RGBALPHA:
        format.setChannels(4);
        break;
    default:
        std::cout<<"warning unknown color type!"<<src.getColorType()<<std::endl;
        break;
    }




    if(src.getBitsPerPixel()==32 && format.getChannels() ==3){
        format.setBitDepth(8);
        format.setChannels(4);
    }else{
        format.setBitDepth(src.getBitsPerPixel()/format.getChannels());
    }


    //    cout << "Channels: " << format.getChannels() << " BitsPerPixel: " << src.getBitsPerPixel() << " Bitdepth: " << format.getBitDepth() << endl;

//    cout << format << endl;

    dest.Format() = format;
    dest.create();
    auto data = src.accessPixels();


    //    if(format.getChannels()==1){
    //        memcpy(dest.getRawData(),data,dest.getSize());
    //    }else if(format.getChannels() == 3 && src.getBitsPerPixel()==24){
    //        memcpy(dest.getRawData(),data,dest.getSize());
    //        dest.flipRB();
    //    }else if(format.getChannels() == 4){
    //        memcpy(dest.getRawData(),data,dest.getSize());
    //    }else{
    //        std::cout<<"TODO: opengl/texture/imageCovnerter.cpp"<<std::endl;
    //        SAIGA_ASSERT(0);
    //    }

    for(int y = 0; y < dest.height; ++y){

        auto targetPtr = dest.positionPtr(0,y);
        auto srcPtr = data + y * src.getScanWidth();
        memcpy(targetPtr,srcPtr,dest.getBytesPerRow());
    }


#if FREEIMAGE_COLORORDER == FREEIMAGE_COLORORDER_BGR
    //convert BGR -> RGB
    if(format.getChannels() == 3 || format.getChannels() == 4){
        dest.flipRB();
    }
#endif

}

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

#endif