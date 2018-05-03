/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/image/imageConverter.h"

#include "saiga/util/assert.h"
#include <cstring> //for memcpy

#ifdef SAIGA_USE_PNG
#include "saiga/image/png_wrapper.h"

namespace Saiga {

void ImageConverter::convert(PNG::PngImage &src, Image& dest)
{
    dest.width = src.width;
    dest.height = src.height;

    dest.type = src.saigaType();


    dest.create();

    for(int i =0; i < dest.rows; ++i)
    {
        memcpy(dest.rowPtr(i),src.rowPtr(i), std::min(dest.pitchBytes,src.bytesPerRow));
    }
}

void ImageConverter::convert(Image &src, PNG::PngImage &dest)
{

    dest.width = src.width;
    dest.height = src.height;

    dest.fromSaigaType(src.type);


    dest.bytesPerRow = iAlignUp(elementSize(src.type)*src.width,4);
    dest.data.resize(dest.bytesPerRow * src.height);

    for(int i =0; i < src.rows; ++i)
    {
        memcpy(dest.rowPtr(i),src.rowPtr(i), std::min(src.pitchBytes,dest.bytesPerRow));
    }
}

}

#endif

#ifdef SAIGA_USE_FREEIMAGE1
#include <FreeImagePlus.h>

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


void ImageConverter::convert(Image src, fipImage &dest){
    if(src.Format().getChannels() == 1 && src.Format().bitsPerPixel()==8){

    }else if(src.Format().getChannels() == 3 && src.Format().bitsPerPixel()==24){
        src.flipRB();
    }else if(src.Format().getChannels() == 4){
        src.flipRB();
    }else{
        std::cout<<"INVALID FORMAT: channels: " << src.Format().getChannels() << ", bitsperpixel " << src.Format().bitsPerPixel() <<std::endl;
        SAIGA_ASSERT(0);
    }



    dest.setSize(getFIT2(src.Format()),src.width,src.height,src.Format().bitsPerPixel());

    auto data = dest.accessPixels();


    memcpy(data,src.getRawData(),src.getSize());


}


void ImageConverter::convert(fipImage &src, Image& dest){
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

    dest.Format() = format;
    dest.create();
    auto data = src.accessPixels();


    if(format.getChannels()==1){
        memcpy(dest.getRawData(),data,dest.getSize());
    }else if(format.getChannels() == 3 && src.getBitsPerPixel()==24){
        memcpy(dest.getRawData(),data,dest.getSize());
        dest.flipRB();
    }else if(format.getChannels() == 4){
        memcpy(dest.getRawData(),data,dest.getSize());
        dest.flipRB();
    }else{
        std::cout<<"TODO: opengl/texture/imageCovnerter.cpp"<<std::endl;
        SAIGA_ASSERT(0);
    }

    //        std::cout << "Image: " << dest.Format() << std::endl;

}
#endif
