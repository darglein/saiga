/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/image/managedImage.h"
#include "saiga/util/assert.h"

//for the load and save function
#include "saiga/image/freeimage.h"
#include "saiga/image/png_wrapper.h"


#include "saiga/util/imath.h"
#include "saiga/image/templatedImage.h"
#include "saiga/util/tostring.h"
#include "saiga/util/color.h"

#include "internal/stb_image_wrapper.h"
#include "internal/stb_image_write_wrapper.h"

#include <fstream>
namespace Saiga {

FileChecker Image::searchPathes;

Image::Image(int h, int w, ImageType type)
    : ImageBase(h,w,iAlignUp(elementSize(type)*w,DEFAULT_ALIGNMENT)), type(type)
{
    create();
}


void Image::create()
{
    SAIGA_ASSERT(width > 0 && height > 0 && type != TYPE_UNKNOWN);

    if(pitchBytes == 0)
    {
        pitchBytes = iAlignUp(elementSize(type)*width,DEFAULT_ALIGNMENT);
    }

    vdata.resize(size());

    SAIGA_ASSERT(valid());
}

void Image::create(int h, int w)
{
    height = h;
    width = w;
    create();
}

void Image::create(int h, int w, ImageType t)
{
    height = h;
    width = w;
    type = t;
    create();

}

void Image::clear()
{
    (*this) = Image();
}

void Image::free()
{
    pitchBytes = 0;
    vdata.clear();
    vdata.shrink_to_fit();
}

void Image::makeZero()
{
    std::fill(vdata.begin(), vdata.end(), 0);
}

bool Image::valid()
{
    return width > 0 && height > 0 && pitchBytes > 0 && type != TYPE_UNKNOWN && size() == vdata.size();
}


std::ostream& operator<<(std::ostream& os, const Image& f)
{
    os << "Image " << f.width << "x" << f.height << " " << " pitch " << f.pitchBytes << " " << " channels/elementType " << channels(f.type) << "/" << elementType(f.type);// << " " << f.Format();
    return os;
}


bool Image::load(const std::string &_path)
{
    clear();


    auto path = searchPathes.getFile(_path);

    bool erg = false;
    std::string type = fileEnding(path);

    if(type == "saigai")
    {
        //saiga raw image format
        loadRaw(path);
    }

#ifdef SAIGA_USE_PNG
    //use libpng for png images
    if(type == "png")
    {
        PNG::PngImage pngimg;
        erg = PNG::readPNG( &pngimg,path);
        if(erg)
            PNG::convert(pngimg,*this);
        return erg;
    }
#endif


    //use libfreeimage if available
#ifdef SAIGA_USE_FREEIMAGE
    erg = FIP::load(path,*this,0);
    return erg;
#endif

    //as a last resort use stb_image.h from the internals directory
    erg = loadImageSTB(path,*this);
    return erg;
}

bool Image::save(const std::string &path)
{
    bool erg = false;
    std::string type = fileEnding(path);

    if(type == "saigai")
    {
        //saiga raw image format
        return saveRaw(path);
    }


#ifdef SAIGA_USE_PNG1
    //use libpng for png images
    if(type == "png")
    {
        PNG::PngImage pngimg;
        PNG::convert(*this,pngimg);
        erg = PNG::writePNG(&pngimg,path);
        return erg;
    }
#endif

#ifdef SAIGA_USE_FREEIMAGE
    erg = FIP::save(path,*this);
    return erg;
#endif

    //as a last resort use stb_image.h from the internals directory
    erg = saveImageSTB(path,*this);
    return erg;
}

#define SAIGA_BINARY_IMAGE_MAGIC_NUMBER 8574385


bool Image::loadRaw(const std::string &path)
{
    clear();
    std::fstream stream;

    try
    {
        stream.open (path,  std::ios::in | std::ios::binary);
    }
    catch (const std::fstream::failure &e)
    {
        std::cout<< e.what() <<std::endl;
        std::cout << "Exception opening/reading file\n";
        return false;
    }

    int magic;
    stream.read((char*)&magic,sizeof(int));
    stream.read((char*)&width,sizeof(int));
    stream.read((char*)&height,sizeof(int));
    stream.read((char*)&type,sizeof(int));
    pitchBytes = 0;

    SAIGA_ASSERT(magic == SAIGA_BINARY_IMAGE_MAGIC_NUMBER);


    create();

    int es = elementSize(type);
    for(int i = 0; i < height; ++i)
    {
        //store it compact
        stream.read((char*)rowPtr(i),width*es);
    }

    stream.close();

    return true;
}

bool Image::saveRaw(const std::string &path)
{
    std::fstream stream;

    try
    {
        stream.open (path,  std::ios::out | std::ios::binary);
    }
    catch (const std::fstream::failure &e)
    {
        std::cout<< e.what() <<std::endl;
        std::cout << "Exception opening/reading file\n";
        return false;
    }

    int magic =8574385;
    stream.write((char*)&magic,sizeof(int));
    stream.write((char*)&width,sizeof(int));
    stream.write((char*)&height,sizeof(int));
    stream.write((char*)&type,sizeof(int));

    int es = elementSize(type);
    for(int i = 0; i < height; ++i)
    {
        //store it compact
        stream.write((char*)rowPtr(i),width*es);
    }
    stream.flush();
    stream.close();

    return true;
}




bool saveHSV(const std::string& path, ImageView<float> img, float vmin, float vmax)
{
    TemplatedImage<float> cpy(img);
    auto vcpy = cpy.getImageView();
    vcpy.add(-vmin);
    vcpy.multWithScalar(float(1) / (vmax-vmin));

    TemplatedImage<ucvec3> simg(img.height,img.width);
    for(int i = 0; i < img.height; ++i)
    {
        for(int j = 0; j < img.width; ++j)
        {
            float f = glm::clamp(vcpy(i,j),0.0f,1.0f);

            //            vec3 hsv = vec3(f,1,1);
            vec3 hsv(f* (240.0/360.0),1,1);
            Saiga::Color c (Color::hsv2rgb(hsv));
            //            unsigned char c = Saiga::iRound(f * 255.0f);
            simg(i,j).r = c.r;
            simg(i,j).g = c.g;
            simg(i,j).b = c.b;
        }
    }
    return simg.save(path);
}


bool save(const std::string& path, ImageView<float> img, float vmin, float vmax)
{
    TemplatedImage<float> cpy(img);
    auto vcpy = cpy.getImageView();

    vcpy.add(-vmin);
    vcpy.multWithScalar(float(1) / (vmax-vmin));

    TemplatedImage<unsigned char> simg(img.height,img.width);
    for(int i = 0; i < img.height; ++i)
    {
        for(int j = 0; j < img.width; ++j)
        {
            float f = glm::clamp(vcpy(i,j),0.0f,1.0f);
            unsigned char c = Saiga::iRound(f * 255.0f);
            simg(i,j) = c;
        }
    }
    return simg.save(path);
}



}
