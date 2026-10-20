/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/util/BinaryFile.h"
#include "saiga/core/util/DataStructures/ArrayView.h"
#include "saiga/core/util/DataStructures/FixedVector.h"
#include "saiga/core/util/DataStructures/ObjectCache.h"
#include "saiga/core/util/DataStructures/ringBuffer.h"
#include "saiga/core/util/Range.h"
#include "saiga/core/util/bitmath.h"
#include "saiga/core/util/color.h"
#include "saiga/core/util/directory.h"
#include "saiga/core/util/discreteProbabilityDistribution.h"
#include "saiga/core/util/encoding.h"
#include "saiga/core/util/env.h"
#include "saiga/core/util/exif/exif.h"
#include "saiga/core/util/fileChecker.h"
#include "saiga/core/util/keyboard.h"
#include "saiga/core/util/perlinnoise.h"
#include "saiga/core/util/statistics.h"
#include "saiga/core/util/string_to_path.h"
#include "saiga/core/util/table.h"
#include "saiga/core/util/tostring.h"
#include "saiga/core/util/translator.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <vector>

using namespace Saiga;

TEST(String, SizeToString)
{
    EXPECT_EQ(sizeToString(512), "512 B");
    EXPECT_EQ(sizeToString(1536), "1536 B");
    EXPECT_EQ(sizeToString(1537), "1.5 kB");
    EXPECT_EQ(sizeToString(3 * 1024 * 1024), "3.0 MB");
    EXPECT_EQ(sizeToString(1500, 1000, 1000, ""), "1.5kB");
}

TEST(String, SizeToStringHuge)
{
    // Sizes of 1536 PiB and more indexed past the end of the unit prefix array.
    EXPECT_EQ(sizeToString(size_t(2000) << 50), "2.0 EB");
    EXPECT_EQ(sizeToString(std::numeric_limits<size_t>::max()), "16.0 EB");
    EXPECT_EQ(sizeToString(std::numeric_limits<size_t>::max(), 1000), "18.4 EB");
}

TEST(Encoding, UTF32toUTF8)
{
    // The lead byte of 3 and 4 byte sequences was 110xxxxx (the 2 byte marker) instead of 1110xxxx / 11110xxx.
    EXPECT_EQ(Encoding::UTF32toUTF8(utf32string{0x41}), "A");
    EXPECT_EQ(Encoding::UTF32toUTF8(utf32string{0xE4}), "\xC3\xA4");
    EXPECT_EQ(Encoding::UTF32toUTF8(utf32string{0x20AC}), "\xE2\x82\xAC");
    EXPECT_EQ(Encoding::UTF32toUTF8(utf32string{0x1F600}), "\xF0\x9F\x98\x80");
}

TEST(Encoding, Roundtrip)
{
    utf32string all;
    for (uint32_t c : {0x0u, 0x7Fu, 0x80u, 0x7FFu, 0x800u, 0xFFFFu, 0x10000u, 0x10FFFFu}) all.push_back(c);
    auto utf8 = Encoding::UTF32toUTF8(all);
    EXPECT_EQ(utf8.size(), 2 * 1 + 2 * 2 + 2 * 3 + 2 * 4);
    EXPECT_EQ(Encoding::UTF8toUTF32(utf8), all);

    std::string text = "a\xC3\xA4\xE2\x82\xAC\xF0\x9F\x98\x80z";
    EXPECT_EQ(Encoding::UTF8toUTF32(text), (utf32string{'a', 0xE4, 0x20AC, 0x1F600, 'z'}));
    EXPECT_EQ(Encoding::UTF32toUTF8(Encoding::UTF8toUTF32(text)), text);
}

TEST(Encoding, TruncatedUTF8)
{
    // A multi byte sequence cut off at the end of the string was read past the end of the string.
    std::string truncated = "ab\xF0\x9F";
    EXPECT_EQ(Encoding::UTF8toUTF32(truncated), (utf32string{'a', 'b'}));
    // Invalid bytes are skipped.
    EXPECT_EQ(Encoding::UTF8toUTF32(std::string("a\x80z")), (utf32string{'a', 'z'}));
}

TEST(Env, SetAndGet)
{
    EXPECT_EQ(SetEnv("SAIGA_TEST_ENV_VALUE", "abc", 1), 0);
    EXPECT_EQ(GetEnv("SAIGA_TEST_ENV_VALUE"), "abc");
    // replace = 0 keeps the old value
    EXPECT_EQ(SetEnv("SAIGA_TEST_ENV_VALUE", "def", 0), 0);
    EXPECT_EQ(GetEnv("SAIGA_TEST_ENV_VALUE"), "abc");
}

TEST(Env, GetMissing)
{
    // getenv returns nullptr for a missing variable, which was passed to the std::string constructor.
    EXPECT_EQ(GetEnv("SAIGA_TEST_ENV_VARIABLE_THAT_IS_NOT_SET"), "");
}

TEST(Color, FloatConstructor)
{
    // The default alpha of the float constructor was 255 instead of 1, which is scaled by 255 again.
    Color c(1.0f, 0.5f, 0.0f);
    EXPECT_EQ(c.r, 255);
    EXPECT_EQ(c.g, 128);
    EXPECT_EQ(c.b, 0);
    EXPECT_EQ(c.a, 255);

    // Out of range values are clamped instead of overflowing the unsigned char.
    Color d(vec4(2.0f, -1.0f, 0.2f, 1.5f));
    EXPECT_EQ(d.r, 255);
    EXPECT_EQ(d.g, 0);
    EXPECT_EQ(d.b, 51);
    EXPECT_EQ(d.a, 255);

    vec4 v = Color(10, 20, 30, 40).toVec4();
    EXPECT_FLOAT_EQ(v(0), 10 / 255.f);
    EXPECT_FLOAT_EQ(v(3), 40 / 255.f);
}

static void ExpectVecNear(const vec3& a, const vec3& b, float eps = 1e-4f)
{
    for (int i = 0; i < 3; ++i) EXPECT_NEAR(a(i), b(i), eps) << i << ": " << a.transpose() << " | " << b.transpose();
}

TEST(Color, HSV)
{
    ExpectVecNear(Color::hsv2rgb(vec3(0, 1, 1)), vec3(1, 0, 0));
    ExpectVecNear(Color::hsv2rgb(vec3(1.f / 3, 1, 1)), vec3(0, 1, 0));
    ExpectVecNear(Color::hsv2rgb(vec3(2.f / 3, 1, 1)), vec3(0, 0, 1));
    ExpectVecNear(Color::hsv2rgb(vec3(0.5f, 0.5f, 0.8f)), vec3(0.4f, 0.8f, 0.8f));

    // rgb2hsv selected the wrong branch of the two GLSL step() functions, e.g. pure green returned v = 0.
    ExpectVecNear(Color::rgb2hsv(vec3(1, 0, 0)), vec3(0, 1, 1));
    ExpectVecNear(Color::rgb2hsv(vec3(0, 1, 0)), vec3(1.f / 3, 1, 1));
    ExpectVecNear(Color::rgb2hsv(vec3(0, 0, 1)), vec3(2.f / 3, 1, 1));
    ExpectVecNear(Color::rgb2hsv(vec3(0.4f, 0.8f, 0.8f)), vec3(0.5f, 0.5f, 0.8f));

    for (float h : {0.05f, 0.2f, 0.45f, 0.6f, 0.8f, 0.95f})
    {
        vec3 hsv(h, 0.7f, 0.9f);
        ExpectVecNear(Color::rgb2hsv(Color::hsv2rgb(hsv)), hsv);
    }
}

TEST(Color, SRGB)
{
    for (float f : {0.0f, 0.001f, 0.2f, 0.5f, 1.0f})
    {
        vec3 c(f, f, f);
        ExpectVecNear(Color::linearrgb2srgb(Color::srgb2linearrgb(c)), c);
        ExpectVecNear(Color::linearrgb2xyz(Color::xyz2linearrgb(c)), c, 1e-3f);
    }
    EXPECT_NEAR(Color::srgb2linearrgb(vec3(0.5f, 0.5f, 0.5f))(0), 0.214f, 1e-3f);
}

static std::filesystem::path UtilTestDir()
{
    auto dir = std::filesystem::temp_directory_path() / "saiga_test_core_util";
    std::filesystem::create_directories(dir);
    return dir;
}

static void WriteUtilText(const std::filesystem::path& file, const std::string& content)
{
    std::ofstream strm(file, std::ios::binary);
    strm << content;
}

TEST(Translator, LastLineWithoutNewline)
{
    // The last line was skipped if the file did not end with a newline.
    auto file = UtilTestDir() / "translation_last_line.txt";
    WriteUtilText(file, "# comment\nhello,hallo,greeting\nyes,ja,");
    Translator t;
    t.init(file.string());
    EXPECT_EQ(t.translate("hello", ""), "hallo");
    EXPECT_EQ(t.translate("yes", ""), "ja");
}

TEST(Translator, EscapedEscapeBeforeSeparator)
{
    // A separator that follows an escaped escape symbol ("\\,") was treated as escaped.
    auto file = UtilTestDir() / "translation_escape.txt";
    WriteUtilText(file, "back\\\\,slash\\\\,\nco\\,mma,Kom\\,ma,\nline,a\\nb,\n");
    Translator t;
    t.init(file.string());
    EXPECT_EQ(t.translate("back\\", ""), "slash\\");
    EXPECT_EQ(t.translate("co,mma", ""), "Kom,ma");
    EXPECT_EQ(t.translate("line", ""), "a\nb");
}

TEST(Translator, WriteReadRoundtrip)
{
    auto file = UtilTestDir() / "translation_roundtrip.txt";
    std::filesystem::remove(file);
    {
        Translator t;
        t.init(file.string());
        // Unknown keys are added with an identity translation.
        EXPECT_EQ(t.translate("plain", "note"), "plain");
        EXPECT_EQ(t.translate("a,b\\c\nd", "x,y"), "a,b\\c\nd");
        EXPECT_EQ(t.translate("ends with\\", ""), "ends with\\");
        t.writeToFile();
    }
    Translator t;
    t.init(file.string());
    EXPECT_EQ(t.translate("plain", ""), "plain");
    EXPECT_EQ(t.translate("a,b\\c\nd", ""), "a,b\\c\nd");
    EXPECT_EQ(t.translate("ends with\\", ""), "ends with\\");
}

// Builds a little endian EXIF segment ("Exif\0\0" + TIFF header). Offsets of the entries are relative to the TIFF
// header, which starts at byte 6.
struct ExifBuilder
{
    std::vector<unsigned char> data;

    ExifBuilder()
    {
        const char header[] = "Exif\0\0II";
        data.insert(data.end(), header, header + 8);
        U16(0x2a);
        U32(8);
    }
    void U16(uint16_t v)
    {
        data.push_back(v & 0xFF);
        data.push_back(v >> 8);
    }
    void U32(uint32_t v)
    {
        U16(v & 0xFFFF);
        U16(v >> 16);
    }
    void Entry(uint16_t tag, uint16_t format, uint32_t length, uint32_t value)
    {
        U16(tag);
        U16(format);
        U32(length);
        U32(value);
    }
    // Offset of the next byte relative to the TIFF header
    uint32_t Offset() const { return uint32_t(data.size() - 6); }
};

TEST(Exif, ParseIFD0)
{
    ExifBuilder b;
    uint32_t make_offset = 8 + 2 + 3 * 12 + 4;
    b.U16(3);
    b.Entry(0x10F, 2, 6, make_offset);  // Make (stored after the IFD)
    b.Entry(0x110, 2, 4, 0x00636261);   // Model "abc\0" stored inline
    b.Entry(0x112, 3, 1, 6);            // Orientation
    b.U32(0);
    ASSERT_EQ(b.Offset(), make_offset);
    const char make[] = "Saiga";
    b.data.insert(b.data.end(), make, make + 6);

    easyexif::EXIFInfo info;
    ASSERT_EQ(info.parseFromEXIFSegment(b.data.data(), b.data.size()), PARSE_EXIF_SUCCESS);
    EXPECT_EQ(info.Make, "Saiga");
    EXPECT_EQ(info.Model, "abc");
    EXPECT_EQ(info.Orientation, 6);
    EXPECT_EQ(info.ByteAlign, 1);

    EXPECT_EQ(info.parseFromEXIFSegment(b.data.data(), 5), PARSE_EXIF_ERROR_NO_EXIF);
    EXPECT_EQ(info.parseFrom(b.data.data(), b.data.size()), PARSE_EXIF_ERROR_NO_JPEG);
}

TEST(Exif, EmptyString)
{
    // A string entry with zero length read the character before the (empty) string.
    ExifBuilder b;
    b.U16(1);
    b.Entry(0x10F, 2, 0, 0);
    b.U32(0);

    easyexif::EXIFInfo info;
    ASSERT_EQ(info.parseFromEXIFSegment(b.data.data(), b.data.size()), PARSE_EXIF_SUCCESS);
    EXPECT_EQ(info.Make, "");
}

TEST(Exif, EmptyRational)
{
    // The focal plane resolution was read from an empty vector if the entry had zero values.
    ExifBuilder b;
    uint32_t sub_ifd = 8 + 2 + 12 + 4;
    b.U16(1);
    b.Entry(0x8769, 4, 1, sub_ifd);
    b.U32(0);
    ASSERT_EQ(b.Offset(), sub_ifd);
    uint32_t value = sub_ifd + 2 + 3 * 12 + 4;
    b.U16(3);
    b.Entry(0xa20e, 5, 0, 0);
    b.Entry(0xa20f, 5, 0, 0);
    b.Entry(0x829d, 5, 1, value);  // FNumber
    b.U32(0);
    ASSERT_EQ(b.Offset(), value);
    b.U32(28);
    b.U32(10);

    easyexif::EXIFInfo info;
    ASSERT_EQ(info.parseFromEXIFSegment(b.data.data(), b.data.size()), PARSE_EXIF_SUCCESS);
    EXPECT_EQ(info.LensInfo.FocalPlaneXResolution, 0);
    EXPECT_EQ(info.LensInfo.FocalPlaneYResolution, 0);
    EXPECT_DOUBLE_EQ(info.FNumber, 2.8);
}

TEST(Exif, GPSOutsideOfSegment)
{
    // The GPS rationals were read without checking the offset against the segment length.
    ExifBuilder b;
    uint32_t gps_ifd = 8 + 2 + 12 + 4;
    b.U16(1);
    b.Entry(0x8825, 4, 1, gps_ifd);
    b.U32(0);
    ASSERT_EQ(b.Offset(), gps_ifd);
    uint32_t lat = gps_ifd + 2 + 2 * 12 + 4;
    b.U16(2);
    b.Entry(2, 5, 3, lat);  // latitude
    b.Entry(6, 5, 1, lat);  // altitude
    b.U32(0);
    ASSERT_EQ(b.Offset(), lat);
    size_t segment_end = b.data.size();
    // Valid rationals behind the end of the segment that is passed to the parser.
    for (int i = 0; i < 3; ++i)
    {
        b.U32(10);
        b.U32(1);
    }

    easyexif::EXIFInfo info;
    ASSERT_EQ(info.parseFromEXIFSegment(b.data.data(), b.data.size()), PARSE_EXIF_SUCCESS);
    EXPECT_NEAR(info.GeoLocation.Latitude, 10 + 10 / 60.0 + 10 / 3600.0, 1e-9);
    EXPECT_EQ(info.GeoLocation.Altitude, 10);

    easyexif::EXIFInfo info2;
    ASSERT_EQ(info2.parseFromEXIFSegment(b.data.data(), segment_end), PARSE_EXIF_SUCCESS);
    EXPECT_EQ(info2.GeoLocation.Latitude, 0);
    EXPECT_EQ(info2.GeoLocation.Altitude, 0);
}

// ============================================================================
// Coverage tests without a known bug
// ============================================================================

TEST(String, SplitConcat)
{
    EXPECT_EQ(split("a,b,,c", ','), (std::vector<std::string>{"a", "b", "", "c"}));
    EXPECT_EQ(split("", ','), (std::vector<std::string>{}));
    EXPECT_EQ(concat({"a", "b", "", "c"}, ','), "a,b,,c");
    EXPECT_EQ(concat({}, ','), "");

    EXPECT_EQ(leadingZeroString(7, 3), "007");
    EXPECT_EQ(leadingZeroString(1234, 3), "1234");

    EXPECT_TRUE(hasEnding("image.png", ".png"));
    EXPECT_FALSE(hasEnding("png", ".png"));
    EXPECT_TRUE(hasPrefix("image.png", "ima"));
    EXPECT_FALSE(hasPrefix("im", "image"));

    EXPECT_EQ(fileEnding("bla.jpg"), "jpg");
    EXPECT_EQ(fileEnding("/usr/local/test.tar.gz"), "gz");
    EXPECT_EQ(removeFileEnding("/usr/local/test.tar.gz"), "/usr/local/test.tar");

    EXPECT_EQ(array_to_string(std::vector<int>{1, 2, 3}, ','), "1,2,3");
    EXPECT_EQ(string_to_array<int>("1 2 3"), (std::vector<int>{1, 2, 3}));
    EXPECT_EQ(string_to_array<double>("0.5,-1", ','), (std::vector<double>{0.5, -1}));
    std::vector<int> v = {4, 5};
    EXPECT_EQ(to_string_iterator(v.begin(), v.end()), "{4, 5}");
    EXPECT_EQ(Saiga::to_string(std::filesystem::path("a/b")), "a/b");
    EXPECT_EQ(Saiga::to_string(0.5, 2), "5.00e-01");
}

TEST(String, StringViewParser)
{
    StringViewParser parser;
    std::string input = "  a, bc,,d\n";
    parser.set(input);
    EXPECT_EQ(parser.next(), "a");
    EXPECT_EQ(parser.next(), "bc");
    EXPECT_EQ(parser.next(), "d");
    EXPECT_EQ(parser.next(), "");

    StringViewParser strict(",", false);
    strict.set("a,,b");
    EXPECT_EQ(strict.next(), "a");
    EXPECT_EQ(strict.next(), "");
    EXPECT_EQ(strict.next(), "b");
}

TEST(String, MatrixToString)
{
    Mat3 M;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) M(i, j) = i * 3 + j + 0.5;
    Mat3 M2;
    StringToMatrix(MatrixToString(M), M2);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) EXPECT_EQ(M(i, j), M2(i, j));
}

TEST(String, Utf8Path)
{
    EXPECT_TRUE(is_valid_utf8("abc"));
    EXPECT_TRUE(is_valid_utf8("\xC3\xA4\xE2\x82\xAC\xF0\x9F\x98\x80"));
    EXPECT_FALSE(is_valid_utf8("\xC3"));              // truncated
    EXPECT_FALSE(is_valid_utf8("\xC0\x80"));          // overlong
    EXPECT_FALSE(is_valid_utf8("\xED\xA0\x80"));      // surrogate
    EXPECT_FALSE(is_valid_utf8("\xF4\x90\x80\x80"));  // > U+10FFFF
    EXPECT_FALSE(is_valid_utf8("\xFF"));

    std::string utf8 = "dir/\xC3\xA4.txt";
    EXPECT_EQ(string_to_path(utf8).u8string(), utf8);
    EXPECT_EQ(path_to_string(std::filesystem::path("a/b.txt")), "a/b.txt");
}

TEST(DataStructures, RingBuffer)
{
    RingBuffer<int> rb(3);
    EXPECT_TRUE(rb.empty());
    EXPECT_EQ(rb.count(), 0);
    rb.add(1);
    rb.add(2);
    rb.add(3);
    EXPECT_TRUE(rb.full());
    EXPECT_EQ(rb.count(), 3);
    EXPECT_EQ(rb.peek(), 1);
    EXPECT_TRUE(rb.addOverride(4));
    EXPECT_EQ(rb.count(), 3);
    EXPECT_EQ(rb.get(), 2);
    EXPECT_FALSE(rb.addOverride(5));
    EXPECT_EQ(rb.get(), 3);
    EXPECT_EQ(rb.get(), 4);
    EXPECT_EQ(rb.count(), 1);
    EXPECT_EQ(rb.get(), 5);
    EXPECT_TRUE(rb.empty());

    // wrap around several times
    for (int i = 0; i < 10; ++i)
    {
        rb.add(i);
        rb.add(i + 100);
        EXPECT_EQ(rb.count(), 2);
        EXPECT_EQ(rb.get(), i);
        EXPECT_EQ(rb.get(), i + 100);
    }
    rb.add(1);
    rb.clear();
    EXPECT_TRUE(rb.empty());
}

namespace
{
struct NonMovable
{
    NonMovable(int v, int* destroyed) : v(v), destroyed(destroyed) {}
    NonMovable(const NonMovable&) = delete;
    ~NonMovable() { (*destroyed)++; }
    int v;
    int* destroyed;
};
}  // namespace

TEST(DataStructures, FixedVector)
{
    int destroyed = 0;
    {
        FixedVectorHeap<NonMovable> heap(4);
        EXPECT_EQ(heap.emplace_back(1, &destroyed), 0);
        EXPECT_EQ(heap.emplace_back(2, &destroyed), 1);
        EXPECT_EQ(heap.size(), 2);
        EXPECT_EQ(heap.back().v, 2);
        EXPECT_TRUE((isAligned<NonMovable, 64>(heap.data())));
        heap.pop_back();
        EXPECT_EQ(destroyed, 1);
        EXPECT_EQ(heap[0].v, 1);
    }
    EXPECT_EQ(destroyed, 2);

    destroyed = 0;
    {
        FixedVectorStack<NonMovable, 3> stack;
        EXPECT_TRUE(stack.empty());
        stack.emplace_back(1, &destroyed);
        stack.emplace_back(2, &destroyed);
        int sum = 0;
        for (auto& e : stack) sum += e.v;
        EXPECT_EQ(sum, 3);
        stack.clear();
        EXPECT_EQ(destroyed, 2);
        stack.emplace_back(3, &destroyed);
    }
    EXPECT_EQ(destroyed, 3);
}

TEST(DataStructures, ArrayView)
{
    std::vector<int> v = {1, 2, 3, 4, 5};
    ArrayView<int> av(v);
    EXPECT_EQ(av.size(), 5);
    EXPECT_EQ(av.byte_size(), 5 * sizeof(int));
    EXPECT_EQ(av.front(), 1);
    EXPECT_EQ(av.back(), 5);
    EXPECT_EQ(av.slice(1, 2).size(), 2);
    EXPECT_EQ(av.slice(1, 2)[0], 2);
    EXPECT_EQ(av.slice_n(3, 2)[1], 5);
    EXPECT_EQ(av.head(2).back(), 2);
    EXPECT_EQ(av.tail(2).front(), 4);
    av.pop_back();
    EXPECT_EQ(av.size(), 4);

    ArrayView<const int> cav = av;
    int sum                  = 0;
    for (auto i : cav) sum += i;
    EXPECT_EQ(sum, 10);

    int arr[3] = {7, 8, 9};
    auto av2   = make_ArrayView(arr);
    EXPECT_EQ(av2.size(), 3);
    av2[1] = 0;
    EXPECT_EQ(arr[1], 0);

    int single = 3;
    EXPECT_EQ(ArrayView<int>(single).size(), 1);
    EXPECT_TRUE(ArrayView<int>().empty());
}

TEST(DataStructures, ObjectCache)
{
    ObjectCache<std::string, int, int> cache;
    cache.put("a", 1, 10);
    cache.put("a", 2, 20);
    int v = 0;
    EXPECT_TRUE(cache.get("a", v, 20));
    EXPECT_EQ(v, 2);
    EXPECT_TRUE(cache.exists("a", 10));
    EXPECT_FALSE(cache.exists("a", 30));
    EXPECT_FALSE(cache.get("b", v, 10));
    cache.clear();
    EXPECT_FALSE(cache.exists("a", 10));
}

TEST(Statistics, Basic)
{
    Statistics<double> st(std::vector<double>{4, 1, 3, 2});
    EXPECT_EQ(st.numValues, 4);
    EXPECT_EQ(st.min, 1);
    EXPECT_EQ(st.max, 4);
    EXPECT_EQ(st.median, 3);
    EXPECT_EQ(st.sum, 10);
    EXPECT_EQ(st.mean, 2.5);
    EXPECT_DOUBLE_EQ(st.variance, 1.25);
    EXPECT_DOUBLE_EQ(st.sdev, std::sqrt(1.25));
    EXPECT_DOUBLE_EQ(st.rms, std::sqrt(30.0 / 4));

    Statistics<float> empty(std::vector<float>{});
    EXPECT_EQ(empty.numValues, 0);
    EXPECT_EQ(empty.mean, 0);
}

TEST(Statistics, Filter)
{
    // gaussianBlurKernel1d/2d need dynamic matrices, which tiny eigen does not have
    std::vector<double> src = {1, 1, 1, 10, 1};
    std::vector<double> dst(src.size());
    applyFilter1D(src, dst, std::vector<double>{0.25, 0.5, 0.25});
    EXPECT_DOUBLE_EQ(dst[0], 1);
    EXPECT_DOUBLE_EQ(dst[2], 3.25);
    EXPECT_DOUBLE_EQ(dst[3], 5.5);
    EXPECT_DOUBLE_EQ(dst[4], 3.25);
}

TEST(Range, Iterate)
{
    std::vector<int> r;
    for (auto i : Range<int>(2, 5)) r.push_back(i);
    EXPECT_EQ(r, (std::vector<int>{2, 3, 4}));

    r.clear();
    for (auto i : Range<int>(0, 6, 2).inverse()) r.push_back(i);
    EXPECT_EQ(r, (std::vector<int>{4, 2, 0}));

    r.clear();
    for (auto i : StridedRange<int>(0, 7, 3)) r.push_back(i);
    EXPECT_EQ(r, (std::vector<int>{0, 3, 6}));

    std::vector<int> keys   = {2, 0};
    std::vector<char> value = {'a', 'b', 'c'};
    std::string s;
    for (auto& c : IndirectRange<char, std::vector<int>::iterator>(keys.begin(), keys.end(), value.data())) s += c;
    EXPECT_EQ(s, "ca");
}

TEST(DiscreteProbabilityDistribution, Frequencies)
{
    Random::setSeed(3487);
    std::vector<double> probabilities = {1, 0, 3, 6};
    DiscreteProbabilityDistribution<double> dist(probabilities);
    std::vector<int> count(4, 0);
    int n = 100000;
    for (int i = 0; i < n; ++i)
    {
        int s = dist.sample();
        ASSERT_GE(s, 0);
        ASSERT_LT(s, 4);
        count[s]++;
    }
    EXPECT_EQ(count[1], 0);
    EXPECT_NEAR(count[0] / double(n), 0.1, 0.01);
    EXPECT_NEAR(count[2] / double(n), 0.3, 0.01);
    EXPECT_NEAR(count[3] / double(n), 0.6, 0.01);
}

TEST(BinaryFile, Roundtrip)
{
    BinaryOutputVector out;
    out << 42 << 1.5 << std::vector<int>{1, 2, 3};

    BinaryInputVector in(out.data.data(), out.data.size());
    int i;
    double d;
    std::vector<int> v;
    in >> i >> d >> v;
    EXPECT_EQ(i, 42);
    EXPECT_EQ(d, 1.5);
    EXPECT_EQ(v, (std::vector<int>{1, 2, 3}));
    EXPECT_EQ(in.current, in.size);

    auto file = std::filesystem::temp_directory_path() / "saiga_test_core_util_binary.bin";
    {
        BinaryFile bf(file, std::ios_base::out);
        bf << 7 << std::vector<float>{0.5f, 2.f};
    }
    BinaryFile bf(file, std::ios_base::in);
    int a;
    std::vector<float> f;
    bf >> a >> f;
    EXPECT_EQ(a, 7);
    EXPECT_EQ(f, (std::vector<float>{0.5f, 2.f}));
}

TEST(BitMath, FloatConversion)
{
    for (unsigned int x : {0u, 1u << 9, 12345u << 9, 0xFFFFFE00u})
    {
        float f = uintToNFloat(x);
        EXPECT_GE(f, 0.f);
        EXPECT_LT(f, 1.f);
        EXPECT_EQ(NFloatToUint(f), x);

        float s = uintToNSFloat(x);
        EXPECT_GE(s, -1.f);
        EXPECT_LT(s, 1.f);
        EXPECT_EQ(NSFloatToUint(s), x);
    }
    EXPECT_EQ(countTrailingZero(1), 0);
    EXPECT_EQ(countTrailingZero(8), 3);
    EXPECT_EQ(countTrailingZero(12), 2);
    EXPECT_EQ(countTrailingZero(1 << 30), 30);
    EXPECT_EQ(countTrailingZero(0), 32);
}

TEST(PerlinNoise, Range)
{
    PerlinNoise reference;
    PerlinNoise seeded(17);
    for (int i = 0; i < 1000; ++i)
    {
        double x = i * 0.137, y = i * 0.071, z = i * 0.023;
        double n = reference.noise(x, y, z);
        EXPECT_GE(n, 0);
        EXPECT_LE(n, 1);
        EXPECT_EQ(n, reference.noise(x, y, z));
        double s = seeded.noise(x, y, z);
        EXPECT_GE(s, 0);
        EXPECT_LE(s, 1);
    }
    // the gradient noise is 0 (-> 0.5) at integer lattice points
    EXPECT_DOUBLE_EQ(reference.noise(3, 5, 7), 0.5);
    EXPECT_GT(reference.fBm(0.3, 0.4, 0.5, 4), 0);
}

TEST(Keyboard, KeyState)
{
    Keyboard k(10);
    EXPECT_EQ(k.getKeyState(3), 0);
    EXPECT_EQ(k.setKeyState(3, 1), 3);
    EXPECT_EQ(k.setKeyState(3, 1), -1);  // unchanged
    EXPECT_EQ(k.getKeyState(3), 1);
    EXPECT_TRUE(k.getMappedKeyState(0, {3}));
    EXPECT_FALSE(k.getMappedKeyState(1, {3}));
    EXPECT_EQ(k.getKeyState(100), 0);
    EXPECT_EQ(k.setKeyState(-1, 1), -1);
}

TEST(Table, Columns)
{
    std::stringstream strm;
    Table table({3, 5}, strm, '|');
    table << "a" << 1 << "bb" << 22;
    EXPECT_EQ(strm.str(), "a  |1    \nbb |22   \n");
}

TEST(FileChecker, SearchPathes)
{
    auto dir = std::filesystem::temp_directory_path() / "saiga_test_core_util_filechecker";
    std::filesystem::create_directories(dir / "sub");
    std::ofstream(dir / "sub" / "file.txt") << "x";
    std::ofstream(dir / "base.txt") << "x";

    FileChecker fc;
    EXPECT_TRUE(fc.getFile("file.txt").empty());
    EXPECT_TRUE(fc.getFile("").empty());
    fc.addSearchPath(dir / "sub");
    EXPECT_EQ(fc.getFile("file.txt"), dir / "sub" / "file.txt");
    EXPECT_EQ(fc("file.txt"), dir / "sub" / "file.txt");
    // relative to the base file first
    EXPECT_EQ(fc.getRelative(dir / "base.txt", "sub/file.txt"), dir / "sub/file.txt");
    EXPECT_EQ(fc.getRelative(dir / "base.txt", "file.txt"), dir / "sub" / "file.txt");
    EXPECT_EQ(FileChecker::getParentDirectory(dir / "base.txt"), dir);
    EXPECT_EQ(fc.getFileName(dir / "base.txt"), "base.txt");

    FileChecker relative;
    relative.addSearchPath(std::filesystem::relative(dir / "sub"));
    relative.setSearchWorkingDirectory(false);
    EXPECT_TRUE(relative.getFile("file.txt").empty());
}

TEST(Directory, Filters)
{
    auto dir = std::filesystem::temp_directory_path() / "saiga_test_core_util_directory";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir / "images_dir");
    for (auto name : {"img_1.png", "img_2.jpg", "other.png"}) std::ofstream(dir / name) << "x";

    Directory d(dir);
    auto sorted = [](std::vector<std::filesystem::path> v)
    {
        std::sort(v.begin(), v.end());
        return v;
    };
    EXPECT_EQ(sorted(d.getFilesEnding(".png")), (std::vector<std::filesystem::path>{"img_1.png", "other.png"}));
    EXPECT_EQ(sorted(d.getFilesPrefix("img_")), (std::vector<std::filesystem::path>{"img_1.png", "img_2.jpg"}));
    EXPECT_EQ(d.getDirectories("_dir"), (std::vector<std::filesystem::path>{"images_dir"}));
    EXPECT_TRUE(d.getDirectories(".png").empty());
    EXPECT_TRUE(d.existsFile("other.png"));
    EXPECT_FALSE(d.existsFile("images_dir"));
}
