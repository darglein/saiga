/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/image/imageBase.h"
#include "saiga/core/image/imageFormat.h"
#include "saiga/core/image/imageView.h"
#include "saiga/core/util/DataStructures/ArrayView.h"

#include <vector>

namespace Saiga
{
#define DEFAULT_ALIGNMENT 4
/**
 * Note: The first scanline is at position data[0].
 */

class SAIGA_CORE_API Image : public ImageBase
{
   public:
    using byte_t   = unsigned char;
    ImageType type = TYPE_UNKNOWN;

   protected:
    std::vector<byte_t> vdata;

   public:
    Image() {}
    Image(ImageType type) : type(type) {}
    Image(int h, int w, ImageType type);
    Image(ImageDimensions dimensions, ImageType type);
    Image(const std::string& file)
    {
        auto res = load(file);
        if (!res)
        {
            SAIGA_EXIT_ERROR("Could not load file " + file);
        }
    }

    // Note: This creates a copy of img
    template <typename T>
    Image(ImageView<T> img)
    {
        setFormatFromImageView(img);
        create();
        img.copyTo(getImageView<T>());
    }

    void create();
    void create(ImageDimensions dimensions);
    void create(int h, int w);
    void create(int h, int w, ImageType t);
    void create(int h, int w, int p, ImageType t);

    void clear();
    void free();
    /**
     * @brief makeZero
     * Sets all data to 0.
     */
    void makeZero();

    /**
     * @brief valid
     * Checks if this image has at least 1 pixel and a valid type.
     */
    bool valid() const;

    void* data() { return vdata.data(); }
    const void* data() const { return vdata.data(); }

    uint8_t* data8() { return vdata.data(); }
    const uint8_t* data8() const { return vdata.data(); }


    template <typename T>
    inline T& at(int y, int x)
    {
        return reinterpret_cast<T*>(rowPtr(y))[x];
    }

    inline void* rowPtr(int y)
    {
        auto ptr = data8() + y * pitchBytes;
        return ptr;
    }

    inline const void* rowPtr(int y) const
    {
        auto ptr = data8() + y * pitchBytes;
        return ptr;
    }


    template <typename T>
    ImageView<T> getImageView()
    {
        SAIGA_ASSERT(ImageTypeTemplate<T>::type == type);
        ImageView<T> res(*this);
        res.data = data();
        return res;
    }

    template <typename T>
    ImageView<const T> getConstImageView() const
    {
        SAIGA_ASSERT(ImageTypeTemplate<T>::type == type);
        ImageView<const T> res(*this);
        res.data = data();
        return res;
    }

    template <typename T>
    void setFormatFromImageView(ImageView<T> v)
    {
        ImageBase::operator=(v);
        type               = ImageTypeTemplate<T>::type;
        pitchBytes         = 0;
    }

    template <typename T>
    void createEmptyFromImageView(ImageView<T> v)
    {
        setFormatFromImageView<T>(v);
        create();
    }


    bool load(const std::string& path);

    // in hint you give a format hint to the decoder
    // Accepted hint values are the types in all lower case.
    // For example: png, jpg, dds,...
    bool loadFromMemory(ArrayView<const char> data, const std::string& hint = "");
    std::vector<unsigned char> saveToMemory(std::string file_extension = "png") const;

    bool save(const std::string& path) const;

    // save in a custom saiga format
    // this can handle all image types
    // If the compress flag is set, we apply zlib lossless compression.
    // Loading dosen't change for compressed files, because we store a flag in the header.
    bool loadRaw(const std::string& path);
    bool saveRaw(const std::string& path, bool compress = false) const;

    /**
     * Tries to convert the given image to a storable format.
     * For example:
     * Floating point images are converted to 8-bit grayscale images.
     */
    bool saveConvert(const std::string& path, float minValue = 0, float maxValue = 1);


    std::vector<uint8_t> compress();
    void decompress(std::vector<uint8_t> data);

    // A glsl-like sample call
    vec4 texture(vec2 uv);

    SAIGA_CORE_API friend std::ostream& operator<<(std::ostream& os, const Image& f);
};



}  // namespace Saiga
