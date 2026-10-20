/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/image/all.h"
#include "saiga/core/util/file.h"

#include "gtest/gtest.h"

#include <cstring>

using namespace Saiga;

static bool LoadFromBytes(Image& img, const std::vector<unsigned char>& bytes, const std::string& hint)
{
    return img.loadFromMemory(ArrayView<const char>((const char*)bytes.data(), bytes.size()), hint);
}

static std::filesystem::path TestDir()
{
    auto dir = std::filesystem::temp_directory_path() / "saiga_test_core_image_decode";
    std::filesystem::create_directories(dir);
    return dir;
}

#ifdef SAIGA_USE_PNG

// 2x2 8 bit gray, Adam7 interlaced, pixels 10 20 / 30 40
static const std::vector<unsigned char> interlaced_png = {
    137, 80,  78, 71, 13, 10,  26,  10,  0,   0,  0,   13, 73, 72, 68, 82, 0,  0,   0,   2,  0,   0,  0,  2,
    8,   0,   0,  0,  1,  32,  218, 98,  110, 0,  0,   0,  15, 73, 68, 65, 84, 120, 156, 99, 224, 98, 16, 97,
    144, 211, 0,  0,  0,  247, 0,   101, 159, 85, 178, 5,  0,  0,  0,  0,  73, 69,  78,  68, 174, 66, 96, 130};

// 2x1 16 bit gray, pixels 0x0102 0xA0B0
static const std::vector<unsigned char> gray16_png = {
    137, 80, 78, 71,  13, 10,  26,  10,  0,   0,  0, 13, 73, 72, 68, 82, 0,  0,   0,   2,  0,  0,   0,  1,
    16,  0,  0,  0,   0,  129, 217, 252, 21,  0,  0, 0,  13, 73, 68, 65, 84, 120, 156, 99, 96, 100, 90, 176,
    1,   0,  1,  255, 1,  84,  142, 125, 142, 64, 0, 0,  0,  0,  73, 69, 78, 68,  174, 66, 96, 130};

TEST(ImageDecode, PngInterlaced)
{
    // Used to abort with an assertion.
    Image img;
    ASSERT_TRUE(LoadFromBytes(img, interlaced_png, "png"));
    ASSERT_EQ(img.type, UC1);
    EXPECT_EQ(img.at<uint8_t>(0, 0), 10);
    EXPECT_EQ(img.at<uint8_t>(0, 1), 20);
    EXPECT_EQ(img.at<uint8_t>(1, 0), 30);
    EXPECT_EQ(img.at<uint8_t>(1, 1), 40);
}

TEST(ImageDecode, Png16Bit)
{
    Image img;
    ASSERT_TRUE(LoadFromBytes(img, gray16_png, "png"));
    ASSERT_EQ(img.type, US1);
    EXPECT_EQ(img.at<uint16_t>(0, 0), 0x0102);
    EXPECT_EQ(img.at<uint16_t>(0, 1), 0xA0B0);

    // Same through the file path
    auto file = TestDir() / "gray16.png";
    File::saveFileBinary(file, gray16_png.data(), gray16_png.size());
    Image img2;
    ASSERT_TRUE(img2.load(file));
    EXPECT_EQ(img2.at<uint16_t>(0, 1), 0xA0B0);
}

TEST(ImageDecode, PngTruncated)
{
    // All cut inside the header or the image data (the IEND chunk at the end is not needed)
    for (size_t size : {0, 8, 30, 40, 50})
    {
        std::vector<unsigned char> bytes(gray16_png.begin(), gray16_png.begin() + size);
        Image img;
        EXPECT_FALSE(LoadFromBytes(img, bytes, "png")) << size;
    }
}

TEST(ImageDecode, SaveUpperCaseExtension)
{
    TemplatedImage<unsigned char> img(3, 4);
    img.makeZero();
    img(1, 2) = 7;
    auto file = TestDir() / "upper.PNG";
    ASSERT_TRUE(img.save(file));
    auto bytes = File::loadFileBinary(file);
    ASSERT_GE(bytes.size(), 8);
    EXPECT_EQ((unsigned char)bytes[1], 'P');
    EXPECT_EQ((unsigned char)bytes[2], 'N');
}
#endif

#ifdef SAIGA_USE_LIBTIFF

struct TiffDesc
{
    uint32_t width = 1, height = 1;
    uint16_t samples = 1, bits = 8, format = 1;
    std::vector<uint8_t> pixels;
    // 0: put the strip right after the header
    uint32_t strip_offset_override = 0;
    uint32_t ifd_offset_override   = 0;
};

// A minimal uncompressed little endian tiff with one strip.
static std::vector<unsigned char> MakeTiff(const TiffDesc& d)
{
    std::vector<unsigned char> out = {'I', 'I', 42, 0, 0, 0, 0, 0};
    auto put16                     = [&](uint16_t v) { out.insert(out.end(), {uint8_t(v), uint8_t(v >> 8)}); };
    auto put32                     = [&](uint32_t v)
    { out.insert(out.end(), {uint8_t(v), uint8_t(v >> 8), uint8_t(v >> 16), uint8_t(v >> 24)}); };

    uint32_t strip_offset = out.size();
    out.insert(out.end(), d.pixels.begin(), d.pixels.end());

    // per sample arrays for BitsPerSample and SampleFormat
    uint32_t bits_offset = out.size();
    for (int i = 0; i < d.samples; ++i) put16(d.bits);
    uint32_t format_offset = out.size();
    for (int i = 0; i < d.samples; ++i) put16(d.format);
    if (out.size() % 2) out.push_back(0);

    uint32_t ifd_offset = out.size();
    struct Entry
    {
        uint16_t tag, type;
        uint32_t count, value;
    };
    auto short_array           = [&](uint32_t offset, uint16_t v) { return d.samples <= 2 ? uint32_t(v) : offset; };
    std::vector<Entry> entries = {
        {256, 4, 1, d.width},
        {257, 4, 1, d.height},
        {258, 3, d.samples, short_array(bits_offset, d.bits)},
        {259, 3, 1, 1},
        {262, 3, 1, uint32_t(d.samples == 1 ? 1 : 2)},
        {273, 4, 1, d.strip_offset_override ? d.strip_offset_override : strip_offset},
        {277, 3, 1, d.samples},
        {278, 4, 1, d.height},
        {279, 4, 1, uint32_t(d.pixels.size())},
        {284, 3, 1, 1},
        {339, 3, d.samples, short_array(format_offset, d.format)},
    };
    put16(entries.size());
    for (auto& e : entries)
    {
        put16(e.tag);
        put16(e.type);
        put32(e.count);
        if (e.type == 3 && e.count <= 2)
        {
            // short values are stored left aligned in the value field
            put16(e.value);
            put16(e.count == 2 ? e.value : 0);
        }
        else
        {
            put32(e.value);
        }
    }
    put32(0);

    uint32_t ifd = d.ifd_offset_override ? d.ifd_offset_override : ifd_offset;
    std::memcpy(&out[4], &ifd, 4);
    return out;
}

TEST(ImageDecode, TiffGray16)
{
    TiffDesc d;
    d.width  = 3;
    d.bits   = 16;
    d.pixels = {1, 0, 2, 0, 0xff, 0xff};
    Image img;
    ASSERT_TRUE(LoadFromBytes(img, MakeTiff(d), "tiff"));
    ASSERT_EQ(img.type, US1);
    EXPECT_EQ(img.at<uint16_t>(0, 0), 1);
    EXPECT_EQ(img.at<uint16_t>(0, 1), 2);
    EXPECT_EQ(img.at<uint16_t>(0, 2), 0xffff);
}

TEST(ImageDecode, TiffRgbLoadsFirstChannel)
{
    // Used to write the full RGB scanline into a single channel image (heap overflow).
    TiffDesc d;
    d.width   = 1000;
    d.height  = 3;
    d.samples = 3;
    for (uint32_t y = 0; y < d.height; ++y)
        for (uint32_t x = 0; x < d.width; ++x) d.pixels.insert(d.pixels.end(), {uint8_t(x + y), 200, 201});

    Image img;
    ASSERT_TRUE(LoadFromBytes(img, MakeTiff(d), "tiff"));
    ASSERT_EQ(img.type, UC1);
    ASSERT_EQ(img.width, 1000);
    for (int y = 0; y < img.height; ++y)
        for (int x = 0; x < img.width; ++x) ASSERT_EQ(img.at<uint8_t>(y, x), uint8_t(x + y)) << x << " " << y;

    // Same through the file path
    auto file  = TestDir() / "rgb.tif";
    auto bytes = MakeTiff(d);
    File::saveFileBinary(file, bytes.data(), bytes.size());
    Image img2;
    ASSERT_TRUE(img2.load(file));
    EXPECT_EQ(img2.at<uint8_t>(2, 999), uint8_t(999 + 2));
}

TEST(ImageDecode, TiffDouble)
{
    // 64 bit floats used to be read into a 32 bit float image (heap overflow).
    TiffDesc d;
    d.width          = 2;
    d.bits           = 64;
    d.format         = 3;
    double values[2] = {1.5, -2.25};
    d.pixels.resize(16);
    std::memcpy(d.pixels.data(), values, 16);
    Image img;
    ASSERT_TRUE(LoadFromBytes(img, MakeTiff(d), "tiff"));
    ASSERT_EQ(img.type, D1);
    EXPECT_EQ(img.at<double>(0, 0), 1.5);
    EXPECT_EQ(img.at<double>(0, 1), -2.25);
}

TEST(ImageDecode, TiffInvalid)
{
    TiffDesc base;
    base.width  = 4;
    base.pixels = {1, 2, 3, 4};
    {
        Image img;
        ASSERT_TRUE(LoadFromBytes(img, MakeTiff(base), "tiff"));
    }

    auto expect_fail = [](const TiffDesc& d, const char* what)
    {
        Image img;
        EXPECT_FALSE(LoadFromBytes(img, MakeTiff(d), "tiff")) << what;
    };

    auto d  = base;
    d.width = 0;
    expect_fail(d, "zero width");

    d      = base;
    d.bits = 12;
    expect_fail(d, "12 bit");

    d      = base;
    d.bits = 64;
    expect_fail(d, "64 bit unsigned");

    // Offsets outside of the data used to move the read pointer outside of the buffer.
    d                     = base;
    d.ifd_offset_override = 0x7ffffff0;
    expect_fail(d, "ifd offset beyond the end");

    d                       = base;
    d.strip_offset_override = 0x7ffffff0;
    expect_fail(d, "strip offset beyond the end");

    auto bytes = MakeTiff(base);
    for (size_t size : {0, 4, 8, 12, 20})
    {
        std::vector<unsigned char> truncated(bytes.begin(), bytes.begin() + size);
        Image img;
        EXPECT_FALSE(LoadFromBytes(img, truncated, "tiff")) << size;
    }
}
#endif

TEST(ImageDecode, SaigaRawRoundtrip)
{
    TemplatedImage<unsigned short> img(5, 7);
    for (int y = 0; y < img.h; ++y)
        for (int x = 0; x < img.w; ++x) img(y, x) = y * 100 + x;

    for (bool compress : {false, true})
    {
        auto file = TestDir() / (compress ? "c.saigai" : "u.saigai");
        ASSERT_TRUE(img.saveRaw(file, compress));
        Image loaded;
        ASSERT_TRUE(loaded.load(file));
        ASSERT_EQ(loaded.type, US1);
        ASSERT_EQ(loaded.h, 5);
        ASSERT_EQ(loaded.w, 7);
        EXPECT_EQ(loaded.at<unsigned short>(4, 6), 406);
    }
}

TEST(ImageDecode, SaigaRawInvalid)
{
    TemplatedImage<unsigned char> img(4, 4);
    img.makeZero();
    auto file = TestDir() / "valid.saigai";
    ASSERT_TRUE(img.saveRaw(file, false));
    auto valid = File::loadFileBinary(file);

    auto expect_fail = [&](std::vector<char> bytes, const char* what)
    {
        auto f = TestDir() / "invalid.saigai";
        File::saveFileBinary(f, bytes.data(), bytes.size());
        Image loaded;
        EXPECT_FALSE(loaded.load(f)) << what;
    };
    auto with_int = [&](int index, int value)
    {
        auto b = valid;
        std::memcpy(&b[index * 4], &value, 4);
        return b;
    };

    expect_fail({}, "empty");
    expect_fail(with_int(0, 1234), "magic");
    expect_fail(with_int(1, 0), "width 0");
    expect_fail(with_int(2, -5), "negative height");
    expect_fail(with_int(1, 1 << 30), "huge width");
    expect_fail(with_int(3, 1000), "type");
    expect_fail(std::vector<char>(valid.begin(), valid.end() - 1), "truncated");

    // compressed data with a wrong size in its header
    ASSERT_TRUE(img.saveRaw(file, true));
    auto compressed = File::loadFileBinary(file);
    auto b          = compressed;
    size_t big      = 1000000;
    std::memcpy(&b[16 + sizeof(size_t)], &big, sizeof(size_t));
    expect_fail(b, "compressed size beyond the end");
    expect_fail(std::vector<char>(compressed.begin(), compressed.begin() + 20), "truncated compressed");
}

TEST(ImageDecode, SaveConvertReturnsResult)
{
    TemplatedImage<float> img(2, 2);
    img.makeZero();
    EXPECT_TRUE(img.saveConvert(TestDir() / "depth.png", 0, 1));
}

TEST(ImageViewOps, InterpolationAtEdges)
{
    TemplatedImage<float> img(2, 3);
    for (int y = 0; y < 2; ++y)
        for (int x = 0; x < 3; ++x) img(y, x) = y * 10 + x;
    auto v = img.getImageView();

    EXPECT_FLOAT_EQ(v.inter(0.5f, 0.5f), 5.5f);
    // The last column/row used to return 0.
    EXPECT_FLOAT_EQ(v.inter(1, 2), 12);
    EXPECT_FLOAT_EQ(v.interUV(1, 1), 12);
    // Outside of the image: clamp to edge instead of extrapolating.
    EXPECT_FLOAT_EQ(v.inter(-3, -3), 0);
    EXPECT_FLOAT_EQ(v.inter(5, 7), 12);
    EXPECT_FLOAT_EQ(v.inter(0, -0.5f), 0);
}

TEST(ImageViewOps, ScaleDownPow2Averages)
{
    TemplatedImage<float> img(4, 4);
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x) img(y, x) = y * 4 + x;
    TemplatedImage<float> small(2, 2);
    img.getImageView().copyScaleDownPow2(small.getImageView(), 2);
    EXPECT_FLOAT_EQ(small(0, 0), (0 + 1 + 4 + 5) / 4.f);
    EXPECT_FLOAT_EQ(small(1, 1), (10 + 11 + 14 + 15) / 4.f);
}

TEST(ImageViewOps, MinMaxAndNormalize)
{
    TemplatedImage<float> img(1, 3);
    img(0, 0) = -5;
    img(0, 1) = -3;
    img(0, 2) = -1;
    float mi, ma;
    img.getImageView().findMinMax(mi, ma);
    EXPECT_EQ(mi, -5);
    EXPECT_EQ(ma, -1);

    img.getImageView().normalize();
    EXPECT_FLOAT_EQ(img(0, 0), 0);
    EXPECT_FLOAT_EQ(img(0, 1), 0.5f);
    EXPECT_FLOAT_EQ(img(0, 2), 1);
}

// Fills all bytes of the visible pixels with a pattern (not the row padding).
static void FillPattern(Image& img, int seed)
{
    int es = elementSize(img.type);
    for (int y = 0; y < img.h; ++y)
    {
        auto row = (uint8_t*)img.rowPtr(y);
        for (int b = 0; b < img.w * es; ++b) row[b] = uint8_t(b * 7 + y * 13 + seed);
    }
}

static bool SamePixels(const Image& a, const Image& b)
{
    if (a.type != b.type || a.dimensions() != b.dimensions()) return false;
    int row_bytes = a.w * elementSize(a.type);
    for (int y = 0; y < a.h; ++y)
    {
        if (std::memcmp(a.rowPtr(y), b.rowPtr(y), row_bytes) != 0) return false;
    }
    return true;
}

static TemplatedImage<ucvec3> SmoothRGB(int h, int w)
{
    TemplatedImage<ucvec3> img(h, w);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) img(y, x) = ucvec3(y * 255 / (h - 1), x * 255 / (w - 1), 128);
    return img;
}

static int MaxAbsDifference(ImageView<const ucvec3> a, ImageView<const ucvec3> b)
{
    int result = 0;
    for (int y = 0; y < a.h; ++y)
        for (int x = 0; x < a.w; ++x)
            result = std::max(result, (a(y, x).cast<int>() - b(y, x).cast<int>()).array().abs().maxCoeff());
    return result;
}

TEST(ImageDecode, SaveRawReportsErrors)
{
    // saveRaw always returned true, even if the file could not be written.
    TemplatedImage<unsigned char> img(2, 2);
    img.makeZero();
    auto file = TestDir() / "does_not_exist" / "x.saigai";
    EXPECT_FALSE(img.saveRaw(file, false));
    EXPECT_FALSE(img.saveRaw(file, true));
    EXPECT_FALSE(img.save(file));
}

TEST(ImageDecode, StbSaveUnsupportedFormat)
{
    // The stb fallback wrote an empty file and returned true for extensions it does not support.
    TemplatedImage<ucvec3> img(2, 2);
    img.makeZero();
    auto file = TestDir() / "unsupported.xyz";
    std::filesystem::remove(file);
    EXPECT_FALSE(img.save(file));

    TemplatedImage<float> fimg(2, 2);
    fimg.makeZero();
    EXPECT_FALSE(fimg.save(TestDir() / "float.jpg"));
}

TEST(ImageDecode, JpgRowPadding)
{
    // The jpg writer ignored the row pitch, so images whose rows are padded (here 13 * 3 = 39 -> 40 bytes) were
    // sheared. Loading jpg data from memory also always failed.
    auto img = SmoothRGB(16, 13);
    ASSERT_NE(img.pitchBytes, 13 * 3);

    auto bytes = img.saveToMemory("jpg");
    ASSERT_FALSE(bytes.empty());
    Image decoded;
    ASSERT_TRUE(LoadFromBytes(decoded, bytes, "jpg"));
    ASSERT_EQ(decoded.type, UC3);
    ASSERT_EQ(decoded.dimensions(), img.dimensions());
    EXPECT_LE(MaxAbsDifference(img.getConstImageView(), decoded.getConstImageView<ucvec3>()), 12);

    auto file = TestDir() / "padded.jpg";
    ASSERT_TRUE(img.save(file));
    Image loaded;
    ASSERT_TRUE(loaded.load(file));
    ASSERT_EQ(loaded.type, UC3);
    EXPECT_LE(MaxAbsDifference(img.getConstImageView(), loaded.getConstImageView<ucvec3>()), 12);
}

TEST(ImageDecode, StbPngCompressRoundtrip)
{
    for (ImageType type : {UC1, UC2, UC3, UC4})
    {
        Image img(5, 7, type);
        FillPattern(img, int(type));
        auto bytes = img.compress();
        ASSERT_FALSE(bytes.empty());
        Image decoded;
        decoded.decompress(bytes);
        EXPECT_TRUE(SamePixels(img, decoded)) << type;
    }
    // Only 8 bit images are supported
    Image img16(2, 2, US1);
    img16.makeZero();
    EXPECT_TRUE(img16.compress().empty());
}
