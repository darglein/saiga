/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/image/image.h"
#include "saiga/util/assert.h"
#include <cstring>
#ifdef SAIGA_USE_FREEIMAGE
#include <FreeImagePlus.h>
#include "saiga/image/freeimage.h"
#endif
#include "saiga/image/png_wrapper.h"
#include "saiga/image/imageConverter.h"
#include "saiga/util/imath.h"
#include "saiga/image/templatedImage.h"
#include "saiga/util/tostring.h"

namespace Saiga {

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
    os << "Image " << f.width << "x" << f.height << " " << " pitch " << f.pitchBytes << " " << " channels/element " << channels(f.type) << "/" << elementType(f.type);// << " " << f.Format();
    return os;
}


bool Image::load(const std::string &path)
{
    bool erg = false;
    std::string type = fileEnding(path);

    if(type == "saigai")
    {
        //saiga raw image format
        SAIGA_ASSERT(0);
    }

#ifdef SAIGA_USE_PNG
    //use libpng for png images
    if(type == "png")
    {
        PNG::PngImage pngimg;
        erg = PNG::readPNG( &pngimg,path);
        if(erg)
            ImageConverter::convert(pngimg,*this);
        return erg;
    }
#endif


    //use libfreeimage if available, libpng otherwise
#ifdef SAIGA_USE_FREEIMAGE
    erg = FIP::load(path,*this,0);
    return erg;
#endif

    // No idea how to save this image
    SAIGA_ASSERT(0);
    return false;
}

bool Image::save(const std::string &path)
{
    bool erg = false;
    std::string type = fileEnding(path);

    if(type == "saigai")
    {
        //saiga raw image format
        SAIGA_ASSERT(0);
    }


#ifdef SAIGA_USE_PNG
    //use libpng for png images
    if(type == "png")
    {
        PNG::PngImage pngimg;
        ImageConverter::convert(*this,pngimg);
        erg = PNG::writePNG(&pngimg,path);
        return erg;
    }
#endif

#ifdef SAIGA_USE_FREEIMAGE
    erg = FIP::save(path,*this);
    return erg;
#endif

    // No idea how to save this image
    SAIGA_ASSERT(0);
    return false;
}



bool saveHSV(const std::string& path, ImageView<float> img, float vmin, float vmax)
{
    std::vector<float> cpy(img.width*img.height);
    ImageView<float> vcpy(img.height,img.width,cpy.data());
    img.copyTo(vcpy);

    vcpy.add(-vmin);
    vcpy.multWithScalar(float(1) / (vmax-vmin));

    TemplatedImage<cvec3> simg(img.height,img.width);
    for(int i = 0; i < img.height; ++i)
    {
        for(int j = 0; j < img.width; ++j)
        {
            float f = glm::clamp(vcpy(i,j),0.0f,1.0f);

            //            vec3 hsv = vec3(f,1,1);
            vec3 hsv(f* (240.0/360.0),1,1);
            Saiga::Color c (Color::hsv2rgb(hsv));
            //            unsigned char c = Saiga::iRound(f * 255.0f);
            simg(j,i).r = c.r;
            simg(j,i).g = c.g;
            simg(j,i).b = c.b;
        }
    }
    return simg.save(path);
}


bool save(const std::string& path, ImageView<float> img, float vmin, float vmax)
{
    std::vector<float> cpy(img.width*img.height);
    ImageView<float> vcpy(img.height,img.width,cpy.data());
    img.copyTo(vcpy);

    vcpy.add(-vmin);
    vcpy.multWithScalar(float(1) / (vmax-vmin));

    TemplatedImage<unsigned char> simg(img.height,img.width);
    for(int i = 0; i < img.height; ++i)
    {
        for(int j = 0; j < img.width; ++j)
        {
            float f = glm::clamp(vcpy(i,j),0.0f,1.0f);
            unsigned char c = Saiga::iRound(f * 255.0f);
            simg(j,i) = c;
        }
    }
    return simg.save(path);
}



}
