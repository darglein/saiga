/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/image/imageBase.h"
#include "saiga/image/imageView.h"
#include "saiga/image/imageFormat.h"
#include "saiga/util/fileChecker.h"
#include "saiga/util/ArrayView.h"
#include <vector>

namespace Saiga {

#define DEFAULT_ALIGNMENT 4
/**
 * Note: The first scanline is at position data[0].
 */

class SAIGA_GLOBAL Image : public ImageBase
{
public:
    //Image search pathes
    static FileChecker searchPathes;

    using byte_t = unsigned char;



    ImageType type = TYPE_UNKNOWN;
protected:
    std::vector<byte_t> vdata;

public:

    Image(){}
    Image(ImageType type) : type(type) {}
    Image(int h, int w , ImageType type);
    Image(std::string file) { auto res = load(file); SAIGA_ASSERT(res); }

    // Note: This creates a copy of img
    template<typename T>
    Image(ImageView<T> img)
    {
        setFormatFromImageView(img);
        create();
        img.copyTo(getImageView<T>());
    }

    void create();
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
    bool valid();

    void* data() { return vdata.data(); }
    const void* data() const { return vdata.data(); }

    uint8_t* data8() { return vdata.data(); }
    const uint8_t* data8() const { return vdata.data(); }


    template<typename T>
    inline
    T& at(int y, int x)
    {
        return reinterpret_cast<T*>(rowPtr(y))[x];
    }

    inline
    void* rowPtr(int y)
    {
        auto ptr = data8() + y * pitchBytes;
        return ptr;
    }


    template<typename T>
    ImageView<T> getImageView()
    {
        SAIGA_ASSERT(ImageTypeTemplate<T>::type == type);
        ImageView<T> res(*this);
        res.data = data();
        return res;
    }

    template<typename T>
    ImageView<const T> getConstImageView() const
    {
        SAIGA_ASSERT(ImageTypeTemplate<T>::type == type);
        ImageView<const T> res(*this);
        res.data = data();
        return res;
    }

    template<typename T>
    void setFormatFromImageView(ImageView<T> v)
    {
        ImageBase::operator=(v);
        type = ImageTypeTemplate<T>::type;
        pitchBytes = 0;
    }

    bool load(const std::string &path);
    bool loadFromMemory(ArrayView<const char> data);

    bool save(const std::string &path);

    // save in a custom saiga format
    // this can handle all image types
    bool loadRaw(const std::string &path);
    bool saveRaw(const std::string &path);


    std::vector<uint8_t> compress();
    void decompress(std::vector<uint8_t> data);

    SAIGA_GLOBAL friend std::ostream& operator<<(std::ostream& os, const Image& f);
};


/**
 * Converts a floating point image to a 8-bit image and saves it.
 * Useful for debugging.
 */
SAIGA_GLOBAL bool saveHSV(const std::string& path, ImageView<float> img, float vmin, float vmax);
SAIGA_GLOBAL bool save(const std::string& path, ImageView<float> img, float vmin, float vmax);



}
