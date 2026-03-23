#define STB_IMAGE_IMPLEMENTATION
#include "stb_image_read_wrapper.h"

#include <iostream>

#include "stb_image.h"
#include "saiga/core/util/string_to_path.h"
#include <fstream>

namespace Saiga
{
bool loadImageSTB(const std::filesystem::path& path, Image& img)
{
    // Handle unicode paths
#ifdef _WIN32
    FILE* file = _wfopen(path.c_str(), L"rb");
#else
    FILE* file = fopen(path.c_str(), "rb");
#endif

    if (!file)
    {
        return false;
    }

    int x, y, n;
    unsigned char* data = stbi_load_from_file(file, &x, &y, &n, 0);
    fclose(file);

    int pitch           = x * n;
    if (!data) return false;

    ImageType it = TYPE_UNKNOWN;

    switch (n)
    {
        case 1:
            it = UC1;
            break;
        case 2:
            it = UC2;
            break;
        case 3:
            it = UC3;
            break;
        case 4:
            it = UC4;
            break;
        default:
            SAIGA_ASSERT(0);
            break;
    }

    img.create(y, x, it);

    for (int i = 0; i < y; ++i)
    {
        auto dst = img.rowPtr(i);
        auto src = data + i * pitch;
        memcpy(dst, src, pitch);
    }

    stbi_image_free(data);
    return true;
}



bool decompressImageSTB(Image& img, std::vector<uint8_t>& _data)
{
    int x, y, n;
    unsigned char* data = stbi_load_from_memory(_data.data(), _data.size(), &x, &y, &n, 0);

    int pitch = x * n;
    if (!data) return false;

    ImageType it = TYPE_UNKNOWN;

    switch (n)
    {
        case 1:
            it = UC1;
            break;
        case 2:
            it = UC2;
            break;
        case 3:
            it = UC3;
            break;
        case 4:
            it = UC4;
            break;
        default:
            SAIGA_ASSERT(0);
            break;
    }

    img.create(y, x, it);

    for (int i = 0; i < y; ++i)
    {
        auto dst = img.rowPtr(i);
        auto src = data + i * pitch;
        memcpy(dst, src, pitch);
    }

    stbi_image_free(data);
    return true;
}

}  // namespace Saiga
