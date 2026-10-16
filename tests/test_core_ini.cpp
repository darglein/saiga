/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/util/file.h"
#include "saiga/core/util/ini/Params.h"
#include "saiga/core/util/tostring.h"

#include "gtest/gtest.h"

#include <clocale>
#include <fstream>

using namespace Saiga;

struct TestParams : public Saiga::ParamsBase
{
    SAIGA_PARAM_STRUCT(TestParams);
    SAIGA_PARAM_STRUCT_FUNCTIONS;

    template <class ParamIterator>
    void Params(ParamIterator* it)
    {
        SAIGA_PARAM(d);
        SAIGA_PARAM(i64);
        SAIGA_PARAM_LIST(v, ' ');
        SAIGA_PARAM_LIST(dl, ',');
        SAIGA_PARAM_LIST(il, ',');
        SAIGA_PARAM_LIST(q, ' ');

        // An 'else' after a param macro must bind to this 'if'.
        if (skip_flag)
            SAIGA_PARAM(flag);
        else
            else_taken = true;
    }

    double d               = 0.1;
    int64_t i64            = 5000000000;
    Vec3 v                 = Vec3(0.1, 1.0 / 3.0, -2e-17);
    std::vector<double> dl = {0.2, 1e300, -3.5};
    std::vector<int> il    = {1, -2, 3};
    Quat q                 = Quat(1, 0, 0, 0);
    bool flag              = false;
    bool skip_flag         = false;
    bool else_taken        = false;
};

static std::filesystem::path TestFile(const std::string& name)
{
    auto dir = std::filesystem::temp_directory_path() / "saiga_test_core_ini";
    std::filesystem::create_directories(dir);
    auto file = dir / name;
    std::filesystem::remove(file);
    return file;
}

static void WriteText(const std::filesystem::path& file, const std::string& content)
{
    std::ofstream strm(file, std::ios::binary);
    strm << content;
}

TEST(Ini, ParseNumber)
{
    double d;
    EXPECT_TRUE(ParseNumber("0.5", d));
    EXPECT_EQ(d, 0.5);
    EXPECT_TRUE(ParseNumber("  -1e-3 ", d));
    EXPECT_EQ(d, -1e-3);
    EXPECT_TRUE(ParseNumber("+2", d));
    EXPECT_EQ(d, 2);
    EXPECT_FALSE(ParseNumber("", d));
    EXPECT_FALSE(ParseNumber("1.5abc", d));
    EXPECT_FALSE(ParseNumber("0,5", d));
    EXPECT_FALSE(ParseNumber("abc", d));

    int i;
    EXPECT_TRUE(ParseNumber("-42", i));
    EXPECT_EQ(i, -42);
    EXPECT_FALSE(ParseNumber("1.5", i));
    EXPECT_FALSE(ParseNumber("99999999999", i));

    long long l;
    EXPECT_TRUE(ParseNumber("5000000000", l));
    EXPECT_EQ(l, 5000000000LL);
}

TEST(Ini, ToDoubleIsLikeAtof)
{
    EXPECT_EQ(to_double("1.5abc"), 1.5);
    EXPECT_EQ(to_double("  2.5"), 2.5);
    EXPECT_EQ(to_double("abc"), 0);
    EXPECT_EQ(to_float("0.25"), 0.25f);
}

TEST(Ini, NumberToStringRoundtrip)
{
    EXPECT_EQ(NumberToStringRoundtrip(0.1), "0.1");
    EXPECT_EQ(NumberToStringRoundtrip(0.1f), "0.1");
    for (double v : {1.0 / 3.0, 0.30000000000000004, -1e-300, 123456789.123456789, 2e300})
    {
        double r;
        ASSERT_TRUE(ParseNumber(NumberToStringRoundtrip(v), r));
        EXPECT_EQ(r, v);
    }
}

TEST(Ini, IndependentOfCLocale)
{
    const char* locales[] = {"de_DE.UTF-8", "de_DE.utf8", "de_DE", "German_Germany.1252"};
    std::string previous  = std::setlocale(LC_NUMERIC, nullptr);
    bool set              = false;
    for (auto l : locales)
    {
        if (std::setlocale(LC_NUMERIC, l))
        {
            set = true;
            break;
        }
    }
    if (!set) GTEST_SKIP() << "No locale with a decimal comma installed";

    double d = 0;
    EXPECT_TRUE(ParseNumber("0.5", d));
    EXPECT_EQ(d, 0.5);
    EXPECT_EQ(to_double("0.25"), 0.25);
    EXPECT_EQ(NumberToStringRoundtrip(0.5), "0.5");

    auto file = TestFile("locale.ini");
    WriteText(file, "[TestParams]\nd = 0.75\nv = 0.5 1.5 2.5\n");
    TestParams p;
    p.Load(file);
    EXPECT_EQ(p.d, 0.75);
    EXPECT_EQ(p.v, Vec3(0.5, 1.5, 2.5));

    std::setlocale(LC_NUMERIC, previous.c_str());
}

TEST(Ini, SaveLoadRoundtrip)
{
    auto file = TestFile("roundtrip.ini");
    TestParams p;
    p.d   = 1.0 / 7.0;
    p.i64 = -7000000000;
    p.v   = Vec3(1.0 / 3.0, 0.1, 1e-200);
    p.dl  = {1.0 / 9.0, -0.0, 5};
    p.il  = {7, 8};
    p.q   = Quat(0.5, 0.5, -0.5, 0.5);
    p.Save(file);

    TestParams p2;
    p2.Load(file);
    EXPECT_EQ(p2.d, p.d);
    EXPECT_EQ(p2.i64, p.i64);
    EXPECT_EQ(p2.v, p.v);
    EXPECT_EQ(p2.dl, p.dl);
    EXPECT_EQ(p2.il, p.il);
    EXPECT_EQ(p2.q.coeffs(), p.q.coeffs());
}

TEST(Ini, DefaultsAreWritten)
{
    // Loading a missing file creates it with the defaults, which must read back exactly.
    auto file = TestFile("defaults.ini");
    TestParams p;
    p.Load(file);
    auto content = File::loadFileString(file);
    EXPECT_NE(content.find("d = 0.1\n"), std::string::npos) << content;

    TestParams p2;
    p2.Load(file);
    TestParams defaults;
    EXPECT_EQ(p2.d, defaults.d);
    EXPECT_EQ(p2.i64, defaults.i64);
    EXPECT_EQ(p2.v, defaults.v);
    EXPECT_EQ(p2.dl, defaults.dl);
}

TEST(Ini, ListWhitespace)
{
    auto file = TestFile("whitespace.ini");
    WriteText(file, "[TestParams]\nv =  1   2 3 \ndl = 1.5, 2.5 ,3\nil = 4 ,5\n");
    TestParams p;
    p.Load(file);
    EXPECT_EQ(p.v, Vec3(1, 2, 3));
    EXPECT_EQ(p.dl, (std::vector<double>{1.5, 2.5, 3}));
    EXPECT_EQ(p.il, (std::vector<int>{4, 5}));
}

TEST(Ini, InvalidValuesThrow)
{
    auto file = TestFile("invalid.ini");
    TestParams p;

    WriteText(file, "[TestParams]\nv = 1 2\n");
    EXPECT_THROW(p.Load(file), std::runtime_error);

    WriteText(file, "[TestParams]\nv = 1 x 3\n");
    EXPECT_THROW(p.Load(file), std::runtime_error);

    WriteText(file, "[TestParams]\ndl = 1,,2\n");
    EXPECT_THROW(p.Load(file), std::runtime_error);

    WriteText(file, "[TestParams]\nil = 1.5\n");
    EXPECT_THROW(p.Load(file), std::runtime_error);

    try
    {
        WriteText(file, "[TestParams]\nv = 1 2\n");
        p.Load(file);
    }
    catch (const std::runtime_error& e)
    {
        EXPECT_NE(std::string(e.what()).find("[TestParams] v"), std::string::npos) << e.what();
    }
}

TEST(Ini, Int64Value)
{
    Saiga::SimpleIni ini;
    ini.SetLongValue("s", "big", 9000000000000000000LL);
    EXPECT_EQ(ini.GetLongValue("s", "big"), 9000000000000000000LL);
    ini.SetLongValue("s", "neg", -5000000000LL);
    EXPECT_EQ(ini.GetLongValue("s", "neg"), -5000000000LL);
    ini.SetLongValue("s", "hex", 0x1234567890LL, nullptr, true);
    EXPECT_EQ(ini.GetLongValue("s", "hex"), 0x1234567890LL);
}

TEST(Ini, ParamMacroElse)
{
    TestParams p;
    p.skip_flag = false;
    Saiga::SimpleIni ini;
    Saiga::IniFileParamIterator it;
    it.ini = &ini;
    p.Params(&it);
    EXPECT_TRUE(p.else_taken);
}

TEST(Ini, CommandLineMatrix)
{
    TestParams p;
    {
        CLI::App app;
        p.Load(app);
        const char* argv[] = {"test", "--TestParams.v", "1.5", "2", "3"};
        EXPECT_NO_THROW(app.parse(5, argv));
        EXPECT_EQ(p.v, Vec3(1.5, 2, 3));
    }
    {
        CLI::App app;
        p.Load(app);
        const char* argv[] = {"test", "--TestParams.v", "1", "x", "3"};
        EXPECT_THROW(app.parse(5, argv), CLI::ParseError);
    }
}
