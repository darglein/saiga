#pragma once


#include "saiga/opengl/opengl.h"
#include "saiga/opengl/texture/imageFormat.h"
#include <stdint.h>
#include <vector>


class SAIGA_GLOBAL Image{
public:

    Image();
    virtual ~Image();
	
    typedef unsigned char byte_t;

    //raw image data
    std::vector<byte_t> data;

    //image dimensions
    int size = 0; //size of data in bytes
    int width = 0;
    int height = 0;
protected:
    ImageFormat format;

    //Alignment of the beginning of each row. Allowed values: 1,2,4,8
    int rowAlignment = 4;
    int bytesPerRow;
public:
    //raw image data
    byte_t* getRawData();
    //byte offset of the given texel in the raw data
    int position(int x, int y);
    //pointer to the beginning of a given texel
    byte_t* positionPtr(int x, int y);


    void setPixel(int x, int y, void* data);
    void setPixel(int x, int y, uint8_t data);
    void setPixel(int x, int y, uint16_t data);
    void setPixel(int x, int y, uint32_t data);

    void setPixel(int x, int y, uint8_t r, uint8_t g, uint8_t b);



    template<typename T>
    T getPixel(int x, int y){
        return *(T*)positionPtr(x,y);
    }


    void makeZero(); //zeros out all bytes
    void create(byte_t* initialData=nullptr);//allocates memory

    //resizes the image. The data is undefined.
    void resize(int w, int h);

    //resizes the image. This adds 0 rows and columns to the left and bottom. The top right is the original image.
    //if the new image is smaller the image is cropped to the new size.
    void resizeCopy(int w, int h);

    void setSubImage(int x, int y, Image &src);
    void setSubImage(int x, int y , int w , int h , uint8_t* data);
    void getSubImage(int x, int y, int w, int h, Image &out);



    void flipRB();

    //======================================================



    size_t getSize();


    ImageFormat& Format();
    const ImageFormat& Format() const;
    int getBytesPerRow() const;
};
