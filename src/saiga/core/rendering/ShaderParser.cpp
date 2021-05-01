/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "ShaderParser.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/util/fileChecker.h"

#include "internal/noGraphicsAPI.h"

#include "ProceduralSkyboxBase.h"

#include <fstream>

namespace Saiga
{
static std::string getFileFromInclude(const std::string& file, std::string line)
{
    const std::string include("#include ");
    line = line.substr(include.size() - 1);

    auto it = std::remove(line.begin(), line.end(), '"');
    line.erase(it, line.end());

    it = std::remove(line.begin(), line.end(), ' ');
    line.erase(it, line.end());

    // recursivly load includes
    std::string includeFileName = line;
    includeFileName             = SearchPathes::shader.getRelative(file, includeFileName);
    return includeFileName;
}


ShaderCode LoadFileAndResolveIncludes(const std::string file, bool add_line_directives)
{
    ShaderCode result;
    result.dependent_files.push_back(file);

    std::ifstream fileStream(file, std::ios::in);
    if (!fileStream.is_open())
    {
        return result;
    }

    const std::string version("#version");
    const std::string include("#include ");

    // TODO: parse with regex. Requires gcc 4.9+

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

        if (include.size() < line.size() && line.compare(0, include.length(), include) == 0)
        {
            std::string includeFileName = getFileFromInclude(file, line);

            if (add_line_directives)
            {
                // add #line before and after #includes
                std::string lineCommand = "#line " + std::to_string(1) + " \"" + includeFileName + "\"";
                ret.push_back(lineCommand);
                addedLines++;
            }
            ret.push_back(line);
            if (add_line_directives)
            {
                std::string lineCommand = "#line " + std::to_string(ret.size() - addedLines + 1) + " \"" + file + "\"";
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
                std::string lineCommand = "#line " + std::to_string(ret.size() - addedLines + 1) + " \"" + file + "\"";
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
        if (include.size() < line.size() && line.compare(0, include.length(), include) == 0)
        {
            std::string includeFileName = getFileFromInclude(file, line);

            auto included_code = LoadFileAndResolveIncludes(includeFileName, add_line_directives);
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
