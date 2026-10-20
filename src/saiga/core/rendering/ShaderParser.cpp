/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "ShaderParser.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/util/assert.h"
#include "saiga/core/util/fileChecker.h"

#include "internal/noGraphicsAPI.h"

#include "ProceduralSkyboxBase.h"

#include <fstream>

namespace Saiga
{
bool ParseShaderInclude(const std::string& line, std::string& file)
{
    file.clear();
    const std::string include("#include");
    if (line.compare(0, include.size(), include) != 0 || line.size() == include.size()) return false;

    auto is_space = [](char c) { return c == ' ' || c == '\t'; };

    size_t pos = include.size();
    if (!is_space(line[pos]) && line[pos] != '"' && line[pos] != '<') return false;
    while (pos < line.size() && is_space(line[pos])) ++pos;
    if (pos == line.size()) return true;

    if (line[pos] == '"' || line[pos] == '<')
    {
        char close = line[pos] == '"' ? '"' : '>';
        auto end   = line.find(close, pos + 1);
        if (end != std::string::npos) file = line.substr(pos + 1, end - pos - 1);
    }
    else
    {
        // Without quotes: the file name ends at whitespace or a comment
        auto end = pos;
        while (end < line.size() && !is_space(line[end]) && line.compare(end, 2, "//") != 0 &&
               line.compare(end, 2, "/*") != 0)
        {
            ++end;
        }
        file = line.substr(pos, end - pos);
    }
    return true;
}

static std::filesystem::path getFileFromInclude(const std::filesystem::path& file, const std::string& include_name)
{
    return SearchPathes::shader.getRelative(file, include_name);
}


ShaderCode LoadFileAndResolveIncludes(const std::filesystem::path& file, bool add_line_directives)
{
    ShaderCode result;
    result.dependent_files.push_back(file);

    std::ifstream fileStream(file, std::ios::in);
    if (!fileStream.is_open())
    {
        return result;
    }

    auto forwardslash_filename = [](const std::filesystem::path& p)
    {
        std::string s = p.u8string();
        std::transform(s.begin(), s.end(), s.begin(),
                       [](char c)
                       {
                           if (c == '\\')
                           {
                               c = '/';
                           }
                           return c;
                       });
        return s;
    };

    const std::string version("#version");

    auto& ret = result.code;
    // first pass:
    // 1. read file line by line and save it into ret vector
    // 2. add #line commands after #version and before and after #includes
    int addedLines = 0;  // count number of added "line" commands.
    while (!fileStream.eof())
    {
        std::string line;
        std::getline(fileStream, line);

        // remove carriage return from windows
        line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());

        std::string include_name;
        if (ParseShaderInclude(line, include_name))
        {
            std::filesystem::path includeFileName =
                include_name.empty() ? std::filesystem::path() : getFileFromInclude(file, include_name);

            if (add_line_directives)
            {
                // add #line before and after #includes
                std::string lineCommand =
                    "#line " + std::to_string(1) + " \"" + forwardslash_filename(includeFileName) + "\"";
                ret.push_back(lineCommand);
                addedLines++;
            }
            ret.push_back(line);
            if (add_line_directives)
            {
                std::string lineCommand =
                    "#line " + std::to_string(ret.size() - addedLines + 1) + " \"" + forwardslash_filename(file) + "\"";
                ret.push_back(lineCommand);
                addedLines++;
            }
        }
        else if (version.size() < line.size() && line.compare(0, version.length(), version) == 0)
        {
            // add a #line command after the #version command
            ret.push_back(line);
            if (add_line_directives)
            {
                std::string lineCommand =
                    "#line " + std::to_string(ret.size() - addedLines + 1) + " \"" + forwardslash_filename(file) + "\"";
                ret.push_back(lineCommand);
                addedLines++;
            }
        }
        else
        {
            ret.push_back(line);
        }
    }

    // second pass:
    // loop over vector and replace #include commands with the actual code

    for (unsigned int i = 0; i < ret.size(); ++i)
    {
        std::string line = ret[i];
        std::string include_name;
        if (ParseShaderInclude(line, include_name))
        {
            std::filesystem::path includeFileName =
                include_name.empty() ? std::filesystem::path() : getFileFromInclude(file, include_name);

            ShaderCode included_code;
            if (!include_name.empty())
            {
                included_code = LoadFileAndResolveIncludes(includeFileName, add_line_directives);
            }
            if (!included_code.valid)
            {
                std::cerr << "ShaderPartLoader: Could not open included file: " << line << std::endl;
                std::cerr << "Extracted filename: '" << includeFileName << "'" << std::endl;
                std::cerr << "Basefile: " << file << std::endl;
                std::cerr << "Make sure it exists and the search pathes are set." << std::endl;
                SAIGA_ASSERT(0);
            }
            result.dependent_files.insert(result.dependent_files.end(), included_code.dependent_files.begin(),
                                          included_code.dependent_files.end());
            ret.erase(ret.begin() + i);
            ret.insert(ret.begin() + i, included_code.code.begin(), included_code.code.end());
        }
    }
    result.valid = true;

    result.DetectParts();
    return result;
}
void ShaderCode::DetectParts()
{
    std::string prefix = "##";

    parts.clear();

    Part current_part;

    for (int i = 0; i < code.size(); ++i)
    {
        auto& line = code[i];
        if (line.compare(0, prefix.size(), prefix) == 0)
        {
            parts.push_back(current_part);
            current_part       = Part();
            current_part.type  = line.substr(prefix.size());
            current_part.start = i + 1;
            current_part.end   = current_part.start;
        }
        else
        {
            current_part.end++;
        }
    }
    parts.push_back(current_part);
}

}  // namespace Saiga
