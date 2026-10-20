/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/util/color.h"
#include "saiga/core/util/encoding.h"
#include "saiga/core/util/env.h"
#include "saiga/core/util/exif/exif.h"
#include "saiga/core/util/tostring.h"
#include "saiga/core/util/translator.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
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
