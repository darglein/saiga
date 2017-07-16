#include "saiga/image/image.h"
#include "saiga/util/assert.h"
#include <cstring>

namespace Saiga {

int Image::getBytesPerRow() const
{
    return bytesPerRow;
}

Image::byte_t *Image::getRawData()
{
    return &data[0];
}

int Image::position(int x, int y){
    return y*bytesPerRow+x*format.bytesPerPixel();
}

Image::byte_t *Image::positionPtr(int x, int y){
    //    return &(this->data[position(x,y)]);
    return getRawData() + position(x,y);
}


void Image::setPixel(int x, int y, void* data){
    std::memcpy(positionPtr(x,y),data,format.bytesPerPixel());
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



void Image::makeZero()
{
    std::fill(data.begin(), data.end(), 0);
}

void Image::create(byte_t* initialData){
    bytesPerRow = width*format.bytesPerPixel();
    int rowPadding = (rowAlignment - (bytesPerRow % rowAlignment)) % rowAlignment;
    bytesPerRow += rowPadding;

    data.resize(getSize());

    if(initialData){
        memcpy(getRawData(),initialData,getSize());
    }
}

void Image::resize(int w, int h)
{
    width = w;
    height = h;
    create();
}

void Image::resizeCopy(int w, int h)
{
    Image oldimg = *this;
    width = w;
    height = h;
    create();
    makeZero();
    setSubImage(0,0,oldimg);
}

void Image::setSubImage(int x, int y, Image& src)
{
    SAIGA_ASSERT(src.width<=width && src.height<=height);
    //copy row by row
    for(int i = 0; i < src.height; i++){
        memcpy(this->positionPtr(x,y+i),src.getRawData()+src.bytesPerRow*i,src.bytesPerRow);
    }
}

void Image::setSubImage(int x, int y, int w, int h, uint8_t *data)
{
    int rowsize = format.bytesPerPixel()*w;
    //copy row by row
    for(int i=0;i<h;i++){
        memcpy(this->positionPtr(x,y+i),data+rowsize*i,rowsize);
    }
}

void Image::getSubImage(int x, int y, int w, int h, Image &out){
    out.width = w;
    out.height = h;
    out.format = format;

    out.create();

    int rowsize = format.bytesPerPixel()*w;

    //copy row by row
    for(int i=0;i<h;i++){//rows
        memcpy(out.getRawData()+rowsize*i,positionPtr(x,y+i),rowsize);
    }


}

void Image::flipRB()
{
    SAIGA_ASSERT(format.getBitDepth()==8 || format.getBitDepth()==16);
    SAIGA_ASSERT(format.getChannels()==3 || format.getChannels()==4);

    if(format.getBitDepth() == 8){
        for(int y = 0 ; y < (int)height ; ++y){
            uint8_t* ptr = getRawData() + (y*bytesPerRow);
            for(int x = 0 ; x < (int)width ; ++x){
                uint8_t r = *ptr;
                *ptr = *(ptr+2);
                *(ptr+2) = r;
                ptr += format.getChannels();
            }
        }
    }else if(format.getBitDepth() == 16){
        for(int y = 0 ; y < (int)height ; ++y){
            uint16_t* ptr = (uint16_t*)(getRawData() + (y*bytesPerRow));
            for(int x = 0 ; x < (int)width ; ++x){
                uint16_t r = *ptr;
                *ptr = *(ptr+2);
                *(ptr+2) = r;
                ptr += format.getChannels();
            }
        }
    }
}

void Image::flipY()
{
    auto copy = data;
    for(int y = 0 ; y < height ; ++y){
        int offset = bytesPerRow * y;
        int invOffset = bytesPerRow * (height-y-1);
        std::copy(copy.begin()+offset,
                  copy.begin()+offset+bytesPerRow,
                  data.begin()+invOffset
                  );
    }
}

void Image::to8bitImage()
{
    if(format.getBitDepth() == 8)
        return;

    SAIGA_ASSERT(format.getElementFormat() == ImageElementFormat::UnsignedNormalized);
    SAIGA_ASSERT(format.getBitDepth() == 16);

    Image old = *this;


    format.setBitDepth(8);
    create();

    for(int y = 0 ; y < height; ++y){
        uint16_t* srcPtr = reinterpret_cast<uint16_t*>(old.positionPtr(0,y));
        uint8_t* targetPtr = reinterpret_cast<uint8_t*>(positionPtr(0,y));
        for(int x = 0 ; x < width * format.getChannels(); ++x){
            targetPtr[x] = int(double(srcPtr[x]) / (65535.0) * 255.0);
        }

    }



}

void Image::removeAlpha()
{
    if(format.getChannels() != 4)
        return;

    Image old = *this;

    format.setChannels(3);
    create();

    for(int y = 0 ; y < height; ++y){
        uint8_t* srcPtr = reinterpret_cast<uint8_t*>(old.positionPtr(0,y));
        uint8_t* targetPtr = reinterpret_cast<uint8_t*>(positionPtr(0,y));
        for(int x = 0 ; x < width; ++x){
            int srcOffset = x * old.Format().bytesPerPixel();
            int targetOffset = x * format.bytesPerPixel();
            for(int i = 0; i < format.bytesPerPixel(); ++ i){
                targetPtr[targetOffset + i] = srcPtr[srcOffset + i];
            }
        }

    }

}

//======================================================


ImageFormat &Image::Format()
{
    return format;
}

const ImageFormat &Image::Format() const
{
    return format;
}



size_t Image::getSize(){
    return height*bytesPerRow;
}


std::ostream& operator<<(std::ostream& os, const Image& f){
    os << "Image " << f.width << "x" << f.height << " " << f.Format();
    return os;
}

}
