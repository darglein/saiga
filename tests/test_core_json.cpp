/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// json11 numbers without the C locale (S23a): parsing through ParseNumber and dump(double) through a classic-locale
// stream, byte-identical to the former "%.17g" output in the C locale.

#include "saiga/core/util/json11.hpp"

#include "gtest/gtest.h"

#include <clocale>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>

using json11::Json;

namespace
{
// The former dump of a number: snprintf("%.17g") in the C locale
std::string printfC(double v)
{
    const std::string previous = std::setlocale(LC_NUMERIC, nullptr);
    std::setlocale(LC_NUMERIC, "C");
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.17g", v);
    std::setlocale(LC_NUMERIC, previous.c_str());
    return buf;
}

double parsed(const std::string& text)
{
    std::string error;
    const Json j = Json::parse(text, error);
    EXPECT_TRUE(error.empty()) << error;
    return j.number_value();
}
}  // namespace

// The dump of a number equals the former "%.17g" output (no JSON writer, MCP reply or test expectation changes)
TEST(Json, DumpMatchesPrintf)
{
    for (double v : {3000000.0, 0.1, 1e300, 1.5, -2.25, 1.0 / 3.0, 123456789.125, 5e-324, -0.0, 1e21, 65536.0})
        EXPECT_EQ(Json(v).dump(), printfC(v)) << v;
    EXPECT_EQ(Json(std::numeric_limits<double>::infinity()).dump(), "null");
}

TEST(Json, ParsesNumbers)
{
    EXPECT_EQ(parsed("1.5"), 1.5);
    EXPECT_EQ(parsed("-0.125"), -0.125);
    EXPECT_EQ(parsed("2e3"), 2000.0);
    EXPECT_EQ(parsed("12345678901234"), 12345678901234.0);
    EXPECT_EQ(parsed("0.1"), 0.1);
    // Beyond the double range (strtod's results)
    EXPECT_EQ(parsed("1e400"), std::numeric_limits<double>::infinity());
    EXPECT_EQ(parsed("-1e400"), -std::numeric_limits<double>::infinity());
    EXPECT_EQ(parsed("1e-400"), 0.0);
}

// Regression (S23a): under a decimal-comma locale strtod read "1.5" as 1 and snprintf wrote "1,5"
TEST(Json, NumbersIndependentOfCLocale)
{
    const char* locales[]      = {"de_DE.UTF-8", "de_DE.utf8", "de_DE", "German_Germany.1252"};
    const std::string previous = std::setlocale(LC_NUMERIC, nullptr);
    bool set                   = false;
    for (auto l : locales)
    {
        if (std::setlocale(LC_NUMERIC, l))
        {
            set = true;
            break;
        }
    }
    if (!set) GTEST_SKIP() << "No locale with a decimal comma installed";

    EXPECT_EQ(parsed("1.5"), 1.5);
    std::string error;
    const Json a = Json::parse("[0.25, -3.75e2, {\"x\": 2.5}]", error);
    EXPECT_TRUE(error.empty()) << error;
    EXPECT_EQ(a[0].number_value(), 0.25);
    EXPECT_EQ(a[1].number_value(), -375.0);
    EXPECT_EQ(a[2]["x"].number_value(), 2.5);
    EXPECT_EQ(Json(1.5).dump(), "1.5");
    for (double v : {3000000.0, 0.1, 1e300}) EXPECT_EQ(Json(v).dump(), printfC(v)) << v;

    std::setlocale(LC_NUMERIC, previous.c_str());
}
