#pragma once



#include <GL/glew.h>
#include <GL/glu.h>


#include <vector>
#include <string>
#include <iostream>
#include <memory>
#include "libhello/util/loader.h"
#include "libhello/util/error.h"
#include "libhello/util/png_wrapper.h"

using std::string;
using std::cout;
using std::endl;


class Image{
public:
    u_int8_t* data = nullptr;
    u_int64_t width, height;
    int bitDepth;
    int channels;
public:

    int bytesPerPixel();
    size_t getSize();
    void setPixel(int x, int y, void* data);
    void setPixel(int x, int y, u_int8_t data);
    void setPixel(int x, int y, u_int16_t data);
    void setPixel(int x, int y, u_int32_t data);

    void setPixel(int x, int y, u_int8_t r, u_int8_t g, u_int8_t b);

    int position(int x, int y);
    u_int8_t* positionPtr(int x, int y);

    void create();//allocates memory
    void createSubImage(int x, int y, int w, int h, Image &out);

    void convertFrom(PNG::Image &image);

    //======================================================

    int getBitDepth() const;
    void setBitDepth(int value);
    int getChannels() const;
    void setChannels(int value);

};
