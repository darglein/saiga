#include "opengl/texture/image.h"
#include <cstring>

int Image::bytesPerPixel(){
    return getChannels()*getBitDepth()/8;
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
    return y*width*bytesPerPixel()+x*bytesPerPixel();
}

uint8_t* Image::positionPtr(int x, int y){
    return this->data+position(x,y);
}

#ifdef USE_PNG
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

#endif

void Image::convertTo(fipImage &fipimg){
    fipimg.setSize(	getFIT(),width,height,bitsPerPixel());

    auto data = fipimg.accessPixels();

    memcpy(data,this->data,getSize());
}

void Image::convertFrom(fipImage &fipimg){
    width = fipimg.getWidth();
    height = fipimg.getHeight();


    switch(fipimg.getColorType()){
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
        cout<<"warning unknown color type!"<<fipimg.getColorType()<<endl;
        break;
    }



    cout<<"create from fipimg: BitsPerPixel "<<fipimg.getBitsPerPixel()<<" channels "<<channels<<" type "<<fipimg.getImageType()<<endl;
    bitDepth = fipimg.getBitsPerPixel()/channels;


    create();


//    RGBQUAD* palette = fipimg.getPalette();
    auto data = fipimg.accessPixels();

    if(channels==1){
        memcpy(this->data,data,getSize());
    }else if(channels == 3){
        for(unsigned int y=0;y<height;++y){
            for(unsigned int x=0;x<width;++x){
                RGBQUAD pixel;
                fipimg.getPixelColor(x,y,&pixel);
                int offset = (y*width+x)*bytesPerPixel();
                this->data[offset] = pixel.rgbRed;
                this->data[offset+1] = pixel.rgbGreen;
                this->data[offset+2] = pixel.rgbBlue;
            }
        }
    }else if(channels == 4){
        for(unsigned int y=0;y<height;++y){
            for(unsigned int x=0;x<width;++x){
                RGBQUAD pixel;
                fipimg.getPixelColor(x,y,&pixel);
                int offset = (y*width+x)*bytesPerPixel();
                this->data[offset] = pixel.rgbRed;
                this->data[offset+1] = pixel.rgbGreen;
                this->data[offset+2] = pixel.rgbBlue;
                this->data[offset+3] =  pixel.rgbReserved;
            }
        }

    }else{
        cout<<"TODO: opengl/texture/image.cpp"<<endl;
    }


    //    if(palette==0){
    //        memcpy(this->data,data,getSize());


    //    }else{
    //        cout<<"palette"<<endl;
    //        for(int i=0;i<getSize();++i){
    //            RGBQUAD& pixel = palette[data[i]];
    //            this->data[i] = pixel.rgbRed;
    //        }
    //    }

}

FREE_IMAGE_TYPE Image::getFIT(){
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

void Image::create(){
    delete[] data;
    data = new uint8_t[getSize()];
}

void Image::createSubImage(int x, int y, int w, int h, Image &out){
    out.width = w;
    out.height = h;
    out.bitDepth = bitDepth;
    out.channels = channels;


    out.create();

    int rowsize = bytesPerPixel()*w;
//    for(int i=y;i<y+h;i++){//rows
//        memcpy(out.data+rowsize*i,data+position(x,y+i),rowsize);
//    }

    for(int i=0;i<h;i++){//rows
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


