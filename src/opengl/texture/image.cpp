#include "opengl/texture/image.h"


int Image::bytesPerPixel(){
    return getChannels()*getBitDepth()/8;
}

size_t Image::getSize(){
    return width*height*bytesPerPixel();
}

void Image::setPixel(int x, int y, void* data){
    memcpy(positionPtr(x,y),data,bytesPerPixel());
}

void Image::setPixel(int x, int y, u_int8_t data){
    *(u_int8_t*)positionPtr(x,y) = data;
}

void Image::setPixel(int x, int y, u_int16_t data){
    *(u_int16_t*)positionPtr(x,y) = data;
}

void Image::setPixel(int x, int y, u_int32_t data){
    *(u_int32_t*)positionPtr(x,y) = data;
}

void Image::setPixel(int x, int y, u_int8_t r, u_int8_t g, u_int8_t b){
    u_int8_t* ptr = positionPtr(x,y);
    ptr[0] = r;
    ptr[1] = g;
    ptr[2] = b;
}

int Image::position(int x, int y){
    return y*width*bytesPerPixel()+x*bytesPerPixel();
}

u_int8_t* Image::positionPtr(int x, int y){
    return this->data+position(x,y);
}

void Image::convertFrom(PNG::Image &image){
    this->width = image.width;
    this->height = image.height;
    this->bitDepth = image.bit_depth;


    switch(image.color_type){
    case PNG_COLOR_TYPE_GRAY:
        this->channels  = 1;
        break;
    case PNG_COLOR_TYPE_GRAY_ALPHA:
        this->channels = 2;
        break;
    case PNG_COLOR_TYPE_RGB:
        this->channels = 3;
        break;
    case PNG_COLOR_TYPE_RGB_ALPHA:
        this->channels = 4;
        break;
    default:
        std::cout<<"Image type not supported: "<<image.color_type<<std::endl;
    }

    cout<<"bits "<<bitDepth<<" channels "<<channels<<endl;

    this->data = image.data;
}

void Image::convertTo(PNG::Image &image){
    image.width = this->width;
    image.height =  this->height;
    image.bit_depth = this->bitDepth;

    switch(this->channels){
    case 1:
        image.color_type = PNG_COLOR_TYPE_GRAY;
        break;
    case 2:
        image.color_type = PNG_COLOR_TYPE_GRAY_ALPHA;
        break;
    case 3:
        image.color_type = PNG_COLOR_TYPE_RGB;
        break;
    case 4:
        image.color_type = PNG_COLOR_TYPE_RGB_ALPHA;
        break;
    default:
        std::cout<<"Image type not supported: "<<this->channels<<std::endl;
    }


    image.data = this->data;
}

void Image::create(){
    delete[] data;
    data = new u_int8_t[getSize()];
}

void Image::createSubImage(int x, int y, int w, int h, Image &out){
    out.width = w;
    out.height = h;
    out.bitDepth = bitDepth;
    out.channels = channels;


    out.create();

    int rowsize = bytesPerPixel()*w;
    for(int i=y;i<y+h;i++){//rows
        memcpy(out.data+rowsize*i,data+position(x,y+i),rowsize);
    }


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
        cout<<"Error Bit Depth not supportet: "<<value<<endl;
        return;
    }
    bitDepth = value;
}


