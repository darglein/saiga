#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write_wrapper.h"

#include <iostream>

#include "stb_image_write.h"

namespace Saiga
{
bool saveImageSTB(const std::filesystem::path& path, const Image& img)
{
    if (img.type == UC1 || img.type == UC2 || img.type == UC3 || img.type == UC4)
    {
        int w               = img.width;
        int h               = img.height;
        int comp            = channels(img.type);
        int stride_in_bytes = img.pitchBytes;

        // Handle unicode paths
#ifdef _WIN32
        FILE* file = _wfopen(path.c_str(), L"wb");
#else
        FILE* file = fopen(path.c_str(), "wb");
#endif

        if (!file)
        {
            return false;
        }


        std::string type = path.extension().string();
        std::transform(type.begin(), type.end(), type.begin(),
            [](char c) { return std::tolower(c); });


        if (type == ".png")
        {
            int len;
            auto data = stbi_write_png_to_mem((unsigned char*)img.data(), stride_in_bytes, w, h, comp, &len);
            fwrite(data, 1, len, file);
        }
        else if (type == ".jpg")
        {
            stbi__write_context s;
            stbi__start_write_callbacks(&s, stbi__stdio_write, (void*)file);
            int r = stbi_write_jpg_core(&s, w, h, comp, (unsigned char*)img.data(), img.get_compression_quality());
        }

        fclose(file);

        return true;
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
