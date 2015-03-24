#pragma once



#include <GL/glew.h>
#include <GL/glu.h>


#include <vector>
#include <string>
#include <iostream>
#include <memory>

//#include "libhello/util/loader.h"
//#include "libhello/util/error.h"
//#include "libhello/util/png_wrapper.h"

#include <FreeImagePlus.h>


using std::string;
using std::cout;
using std::endl;


class Image{
public:

    ~Image();
	
	uint8_t* data = nullptr;
    uint64_t width, height;
    int bitDepth;
    int channels;
public:

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

    void create();//allocates memory
    void createSubImage(int x, int y, int w, int h, Image &out);

#ifdef USE_PNG
    void convertFrom(PNG::Image &image);
    void convertTo(PNG::Image &image);
#endif

    void convertTo(fipImage &fipimg);
    void convertFrom(fipImage &fipimg);

    FREE_IMAGE_TYPE getFIT();

    //======================================================

    int getBitDepth() const;
    void setBitDepth(int value);
    int getChannels() const;
    void setChannels(int value);

};
