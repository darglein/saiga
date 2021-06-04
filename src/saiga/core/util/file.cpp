/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "file.h"

#include "saiga/core/util/assert.h"

#include "internal/noGraphicsAPI.h"

#include <algorithm>
#include <fstream>
#include <iostream>

namespace Saiga
{
namespace File
{
std::vector<char> loadFileBinary(const std::string& file)
{
    std::vector<char> result;

    std::ifstream is(file, std::ios::binary | std::ios::in | std::ios::ate);
    if (!is.is_open())
    {
        std::cout << "File not found " << file << std::endl;
        return result;
    }

    size_t size = is.tellg();
    result.resize(size);
    is.seekg(0, std::ios::beg);
    is.read(result.data(), size);
    is.close();
    return result;
}

std::string loadFileString(const std::string& file)
{
    std::string fileContent;
    std::ifstream fileStream(file, std::ios::in);
    if (!fileStream.is_open())
    {
        std::cout << "File not found " << file << std::endl;
        return "";
    }
    std::string line = "";
    while (!fileStream.eof())
    {
        getline(fileStream, line);
        fileContent.append(line + "\n");
    }
    fileStream.close();

    if (fileContent.size() >= 3)
    {
        // remove utf8 bom
        const unsigned char* data = (const unsigned char*)fileContent.data();
        if (data[0] == 0XEF && data[1] == 0XBB && data[2] == 0XBF)
        {
            fileContent.erase(0, 3);
        }
    }

    return fileContent;
}

std::vector<std::string> loadFileStringArray(const std::string& file)
{
    std::vector<std::string> fileContent;
    std::ifstream fileStream(file, std::ios::in);
    if (!fileStream.is_open())
    {
        std::cout << "File not found " << file << std::endl;
        return {};
    }
    std::string line = "";
    while (!fileStream.eof())
    {
        getline(fileStream, line);
        fileContent.push_back(line);
    }
    fileStream.close();

    if (fileContent.size() > 0)
    {
        auto& fc = fileContent.front();
        if (fc.size() >= 3)
        {
            // remove utf8 bom
            const unsigned char* data = (const unsigned char*)fc.data();
            if (data[0] == 0XEF && data[1] == 0XBB && data[2] == 0XBF)
            {
                fc.erase(0, 3);
            }
        }
    }

    return fileContent;
}

void saveFileBinary(const std::string& file, const void* data, size_t size)
{
    std::ofstream is(file, std::ios::binary | std::ios::out);
    if (!is.is_open())
    {
        std::cout << "File not found " << file << std::endl;
        return;
    }

    is.write((const char*)data, size);
    is.close();
}

void removeWindowsLineEnding(std::string& line)
{
    line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
}


void removeWindowsLineEnding(std::vector<std::string>& file)
{
    for (auto& line : file)
    {
        removeWindowsLineEnding(line);
    }
}



}  // namespace File
}  // namespace Saiga
