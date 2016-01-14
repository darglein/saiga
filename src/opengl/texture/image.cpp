#include "saiga/opengl/texture/image.h"
#include <cstring>
#include <iostream>
#include <FreeImagePlus.h>

#ifdef USE_PNG
    #include "saiga/util/png_wrapper.h"
#endif


Image::Image()
{
}

Image::~Image()
{
    if(shouldDelete)
        delete[] data;
}

int Image::bytesPerChannel(){
    return getBitDepth()/8;
}

int Image::bytesPerPixel(){
    return getChannels()*bytesPerChannel();
}

int Image::bitsPerPixel(){
    return getChannels()*getBitDepth();
}

size_t Image::getSize(){
    return width*height*bytesPerPixel();
}

void Image::setPixel(int x, int y, void* data){
    std::memcpy(positionPtr(x,y),data,bytesPerPixel());
}

void Image::setPixel(int x, int y, uint8_t data){
    *(uint8_t*)positionPtr(x,y) = data;
}

void Image::setPixel(int x, int y, uint16_t data){
    *(uint16_t*)positionPtr(x,y) = data;
}

void Image::setPixel(int x, int y, uint32_t data){
    *(uint32_t*)positionPtr(x,y) = data;
}

void Image::setPixel(int x, int y, uint8_t r, uint8_t g, uint8_t b){
    uint8_t* ptr = positionPtr(x,y);
    ptr[0] = r;
    ptr[1] = g;
    ptr[2] = b;
}

int Image::position(int x, int y){
    return (y*width+x)*bytesPerPixel();
}

uint8_t* Image::positionPtr(int x, int y){
    return this->data+position(x,y);
}

void Image::makeZero()
{
    memset(data,0,getSize());
}

void Image::create(){

    delete[] data;
    data = new uint8_t[getSize()];

    shouldDelete = true;
}

void Image::setSubImage(int x, int y, int w, int h, uint8_t *data)
{
    int rowsize = bytesPerPixel()*w;
    for(int i=0;i<h;i++){//rows
        memcpy(this->data+position(x,y+i),data+rowsize*i,rowsize);
    }
}

void Image::getSubImage(int x, int y, int w, int h, Image &out){
    out.width = w;
    out.height = h;
    out.bitDepth = bitDepth;
    out.channels = channels;
    out.srgb = srgb;


    out.create();

    int rowsize = bytesPerPixel()*w;

    for(int i=0;i<h;i++){//rows
        memcpy(out.data+rowsize*i,data+position(x,y+i),rowsize);
    }


}

void Image::addChannel()
{
    auto oldData = data;
    data = nullptr;
    int oldBpp = bytesPerPixel();


    this->channels++;
    this->create();

    int newBpp = bytesPerPixel();

    for(int y = 0 ; y < (int)height ; ++y){
        for(int x = 0 ; x < (int)width ; ++x){
            int pos = y * width + x;
            auto posOld = oldData + pos * oldBpp;
            auto posNew = data + pos * newBpp;

            for(int i = 0 ;i < newBpp ; ++i){
                posNew[i] = (i<oldBpp)?posOld[i] : 0;
            }
        }
    }

    delete[] oldData;
}

//======================================================

int Image::getChannels() const
{
    return channels;
}

void Image::setChannels(int value)
{
    channels = value;
}

int Image::getBitDepth() const
{

    return bitDepth;
}

void Image::setBitDepth(int value)
{
    if(value%8!=0){
        std::cout<<"Error Bit Depth not supportet: "<<value<<std::endl;
        return;
    }
    bitDepth = value;
}


