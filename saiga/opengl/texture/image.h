#pragma once


#include "saiga/opengl/opengl.h"


#include <stdint.h>

class fipImage;

namespace PNG{
class Image;
}

class SAIGA_GLOBAL Image{
public:

    Image();
    ~Image();
	
	uint8_t* data = nullptr;
    uint64_t width, height;
    int bitDepth;
    int channels;
    bool srgb = false;
    bool shouldDelete = false;
public:
    int bytesPerChannel();
    int bytesPerPixel();
    int bitsPerPixel();
    size_t getSize();
    void setPixel(int x, int y, void* data);
    void setPixel(int x, int y, uint8_t data);
    void setPixel(int x, int y, uint16_t data);
    void setPixel(int x, int y, uint32_t data);

    void setPixel(int x, int y, uint8_t r, uint8_t g, uint8_t b);

    int position(int x, int y);
    uint8_t* positionPtr(int x, int y);

    template<typename T>
    T getPixel(int x, int y){
        return *(T*)positionPtr(x,y);
    }

    void makeZero(); //zeros out all bytes
    void create();//allocates memory

    void setSubImage(int x, int y , int w , int h , uint8_t* data);
    void getSubImage(int x, int y, int w, int h, Image &out);

    //adds a zero initialized channel
    void addChannel();


    //======================================================

    int getBitDepth() const;
    void setBitDepth(int value);
    int getChannels() const;
    void setChannels(int value);


};
