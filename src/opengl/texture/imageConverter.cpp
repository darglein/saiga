#include "saiga/opengl/texture/imageConverter.h"
#include <iostream>
#include "saiga/util/assert.h"
#include <cstring> //for memcpy

#ifdef USE_PNG
#include "saiga/util/png_wrapper.h"

void ImageConverter::convert(PNG::Image &src, Image& dest){
    dest.width = src.width;
    dest.height = src.height;
    dest.bitDepth = src.bit_depth;


    switch(src.color_type){
    case PNG_COLOR_TYPE_GRAY:
        dest.channels  = 1;
        break;
    case PNG_COLOR_TYPE_GRAY_ALPHA:
        dest.channels = 2;
        break;
    case PNG_COLOR_TYPE_RGB:
        dest.channels = 3;
        break;
    case PNG_COLOR_TYPE_RGB_ALPHA:
        dest.channels = 4;
        break;
    default:
        std::cout<<"Image type not supported: "<<src.color_type<<std::endl;
    }

//    std::cout<<"bits "<<bitDepth<<" channels "<<channels<<std::endl;
    dest.create(src.data);
//    dest.data = src.data;
}

void ImageConverter::convert(Image& src, PNG::Image &dest){
    dest.width = src.width;
    dest.height =  src.height;
    dest.bit_depth = src.bitDepth;

    switch(src.channels){
    case 1:
        dest.color_type = PNG_COLOR_TYPE_GRAY;
        break;
    case 2:
        dest.color_type = PNG_COLOR_TYPE_GRAY_ALPHA;
        break;
    case 3:
        dest.color_type = PNG_COLOR_TYPE_RGB;
        break;
    case 4:
        dest.color_type = PNG_COLOR_TYPE_RGB_ALPHA;
        break;
    default:
        std::cout<<"Image type not supported: "<<src.channels<<std::endl;
    }


    dest.data = src.getRawData();
}

#endif

#ifdef USE_FREEIMAGE
#include <FreeImagePlus.h>

FREE_IMAGE_TYPE getFIT2(int bitDepth, int channels){
    if(bitDepth==16 && channels==3){
        return FIT_RGB16;
    }else if(bitDepth==16 && channels==4){
        return FIT_RGBA16;
    }else if(bitDepth==16 && channels==1){
        return FIT_UINT16;
    }else if(bitDepth==32 && channels==1){
        return FIT_UINT32;
    }

    return FIT_BITMAP;
}


void ImageConverter::convert(Image& src, fipImage &dest){
    dest.setSize(getFIT2(src.bitDepth,src.channels),src.width,src.height,src.bitsPerPixel());

    //free image pads lines to 4 bytes
//    int scanWidth = dest.getScanWidth();

    auto data = dest.accessPixels();
//    for(int y = 0 ; y < src.height ; ++y){

//        auto rowPtr = src.positionPtr(0,y);
//        memcpy(data+scanWidth*y,rowPtr,scanWidth);
//    }

    memcpy(data,src.getRawData(),src.getSize());

}


void ImageConverter::convert(fipImage &src, Image& dest){
    dest.width = src.getWidth();
    dest.height = src.getHeight();


    switch(src.getColorType()){
    case FIC_MINISBLACK:
        dest.channels = 1;
        break;
    case FIC_RGB:
        dest.channels = 3;
        break;
    case FIC_RGBALPHA:
        dest.channels = 4;
        break;
    default:
        std::cout<<"warning unknown color type!"<<src.getColorType()<<std::endl;
        break;
    }


    dest.bitDepth = src.getBitsPerPixel()/dest.channels;

    if(src.getBitsPerPixel()==32 && dest.channels ==3){
        dest.bitDepth = 8;
        dest.channels = 4;
    }

    dest.create();
    auto data = src.accessPixels();


    if(dest.channels==1){
        memcpy(dest.getRawData(),data,dest.getSize());
    }else if(dest.channels == 3 && src.getBitsPerPixel()==24){
        memcpy(dest.getRawData(),data,dest.getSize());
        dest.flipRB();
    }else if(dest.channels == 4){
        memcpy(dest.getRawData(),data,dest.getSize());
        dest.flipRB();
    }else{
        std::cout<<"TODO: opengl/texture/imageCovnerter.cpp"<<std::endl;
        assert(0);
    }

}
#endif
