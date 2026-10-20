/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/rendering/ShaderParser.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <filesystem>
#include <fstream>

using namespace Saiga;

static void WriteFile(const std::filesystem::path& file, const std::string& content)
{
    std::filesystem::create_directories(file.parent_path());
    std::ofstream strm(file);
    strm << content;
}

static bool Contains(const std::vector<std::string>& code, const std::string& line)
{
    return std::find(code.begin(), code.end(), line) != code.end();
}

TEST(ShaderParser, ParseShaderInclude)
{
    std::string file;
    EXPECT_TRUE(ParseShaderInclude("#include \"a.glsl\"", file));
    EXPECT_EQ(file, "a.glsl");
    EXPECT_TRUE(ParseShaderInclude("#include \"dir/my file.glsl\" // \"comment\"", file));
    EXPECT_EQ(file, "dir/my file.glsl");
    EXPECT_TRUE(ParseShaderInclude("#include\t<b.glsl>\t", file));
    EXPECT_EQ(file, "b.glsl");
    EXPECT_TRUE(ParseShaderInclude("#include\"c.glsl\"", file));
    EXPECT_EQ(file, "c.glsl");
    EXPECT_TRUE(ParseShaderInclude("#include d.glsl// comment", file));
    EXPECT_EQ(file, "d.glsl");
    EXPECT_TRUE(ParseShaderInclude("#include e.glsl /* comment */", file));
    EXPECT_EQ(file, "e.glsl");

    // Malformed include directives
    EXPECT_TRUE(ParseShaderInclude("#include \"f.glsl", file));
    EXPECT_EQ(file, "");
    EXPECT_TRUE(ParseShaderInclude("#include   ", file));
    EXPECT_EQ(file, "");

    // No include directives
    EXPECT_FALSE(ParseShaderInclude("#include", file));
    EXPECT_FALSE(ParseShaderInclude("#includes \"a.glsl\"", file));
    EXPECT_FALSE(ParseShaderInclude("// #include \"a.glsl\"", file));
    EXPECT_FALSE(ParseShaderInclude("#version 330", file));
    EXPECT_FALSE(ParseShaderInclude("", file));
}

// The include parser only removed quotes and spaces, so a trailing comment, a tab or <...> became part of the file
// name and the include was not found.
TEST(ShaderParser, IncludeSyntax)
{
    auto dir = std::filesystem::temp_directory_path() / "saiga_test_core_shader_parser";
    std::filesystem::remove_all(dir);

    WriteFile(dir / "main.glsl",
              "##GL_FRAGMENT_SHADER\n"
              "#version 330\n"
              "#include \"a.glsl\" // a comment\n"
              "#include\t\"b.glsl\"\t\n"
              "#include <c.glsl>\n"
              "#include  \"d.glsl\"/* another comment */\n"
              "void main() {}\n");
    WriteFile(dir / "a.glsl", "const int A = 1;\n");
    WriteFile(dir / "b.glsl", "const int B = 2;\n");
    WriteFile(dir / "c.glsl", "const int C = 3;\n");
    WriteFile(dir / "d.glsl", "const int D = 4;\n");

    auto code = LoadFileAndResolveIncludes(dir / "main.glsl", false);
    ASSERT_TRUE(code.valid);
    EXPECT_EQ(code.dependent_files.size(), 5u);
    EXPECT_TRUE(Contains(code.code, "const int A = 1;"));
    EXPECT_TRUE(Contains(code.code, "const int B = 2;"));
    EXPECT_TRUE(Contains(code.code, "const int C = 3;"));
    EXPECT_TRUE(Contains(code.code, "const int D = 4;"));
    for (auto& line : code.code) EXPECT_EQ(line.find("#include"), std::string::npos) << line;

    std::filesystem::remove_all(dir);
}
