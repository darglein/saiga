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
