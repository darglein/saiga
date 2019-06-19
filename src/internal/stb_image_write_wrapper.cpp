#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write_wrapper.h"

#include <iostream>

#include "stb_image_write.h"

namespace Saiga
{
bool saveImageSTB(const std::string& path, const Image& img)
{
    if (img.type == UC1 || img.type == UC2 || img.type == UC3 || img.type == UC4)
    {
        int w               = img.width;
        int h               = img.height;
        int comp            = channels(img.type);
        int stride_in_bytes = img.pitchBytes;

        auto res = stbi_write_png(path.c_str(), w, h, comp, img.data(), stride_in_bytes);
        return res != 0;
    }
    else
    {
        std::cerr << "saveImageSTB: unsupported image type." << std::endl;
        return false;
    }
}

std::vector<uint8_t> compressImageSTB(const Image& img)
{
    int w               = img.width;
    int h               = img.height;
    int comp            = channels(img.type);
    int stride_in_bytes = img.pitchBytes;


    int len;
    unsigned char* png = stbi_write_png_to_mem((unsigned char*)img.data(), stride_in_bytes, w, h, comp, &len);

    std::vector<uint8_t> data(len);
    memcpy(data.data(), png, len);


    STBIW_FREE(png);

    return data;
}



}  // namespace Saiga
