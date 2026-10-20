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
