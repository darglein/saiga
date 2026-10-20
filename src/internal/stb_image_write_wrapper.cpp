#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write_wrapper.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>

#include "stb_image_write.h"

namespace Saiga
{
static void AppendToVector(void* context, void* data, int size)
{
    auto vec   = static_cast<std::vector<uint8_t>*>(context);
    auto bytes = static_cast<const uint8_t*>(data);
    vec->insert(vec->end(), bytes, bytes + size);
}

// Encodes an 8 bit image as "png" or "jpg". Returns an empty vector if the type or format is not supported or the
// encoding failed.
static std::vector<uint8_t> EncodeImageSTB(const Image& img, const std::string& type)
{
    if (img.type != UC1 && img.type != UC2 && img.type != UC3 && img.type != UC4)
    {
        return {};
    }

    int w               = img.width;
    int h               = img.height;
    int comp            = channels(img.type);
    int stride_in_bytes = img.pitchBytes;

    std::vector<uint8_t> result;
    int ok = 0;
    if (type == "png")
    {
        ok = stbi_write_png_to_func(AppendToVector, &result, w, h, comp, img.data(), stride_in_bytes);
    }
    else if (type == "jpg")
    {
        // The jpg writer has no stride parameter, so the rows must be tightly packed.
        size_t row_bytes = size_t(w) * comp;
        std::vector<uint8_t> packed(row_bytes * h);
        for (int y = 0; y < h; ++y)
        {
            memcpy(packed.data() + y * row_bytes, img.rowPtr(y), row_bytes);
        }
        ok = stbi_write_jpg_to_func(AppendToVector, &result, w, h, comp, packed.data(), img.get_compression_quality());
    }

    if (!ok)
    {
        result.clear();
    }
    return result;
}

bool saveImageSTB(const std::filesystem::path& path, const Image& img)
{
    std::string type = path.extension().string();
    std::transform(type.begin(), type.end(), type.begin(), [](char c) { return std::tolower(c); });
    if (!type.empty() && type[0] == '.') type = type.substr(1);

    auto data = EncodeImageSTB(img, type);
    if (data.empty())
    {
        std::cerr << "saveImageSTB: unsupported image type or file format " << path << std::endl;
        return false;
    }

    std::ofstream file(path, std::ios::binary | std::ios::out);
    file.write((const char*)data.data(), data.size());
    file.close();
    return bool(file);
}

std::vector<uint8_t> compressImageSTB(const Image& img, const std::string& type)
{
    return EncodeImageSTB(img, type);
}



}  // namespace Saiga
