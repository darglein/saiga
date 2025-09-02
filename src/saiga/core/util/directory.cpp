/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/util/directory.h"

#include "saiga/core/util/assert.h"
#include "saiga/core/util/tostring.h"

#include "internal/noGraphicsAPI.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <sys/stat.h>

namespace Saiga
{
Directory::Directory(const std::filesystem::path& dir)
{
    dirname = dir;
}

Directory::~Directory()
{
}

std::vector<std::filesystem::path> Directory::getFiles()
{
    std::vector<std::filesystem::path> out;

    for (auto& it : std::filesystem::directory_iterator(dirname))
    {
        if (it.is_regular_file())
        {
            out.push_back(std::filesystem::relative(it.path(), dirname));
        }
    }
    return out;
}


std::vector<std::filesystem::path> Directory::getFilesEnding(const std::filesystem::path& ending)
{
    auto tmp = getFiles();
    auto e   = std::remove_if(tmp.begin(), tmp.end(), [&](const std::filesystem::path& s) { return !hasEnding(s.u8string(), ending.u8string()); });
    tmp.erase(e, tmp.end());
    return tmp;
}

std::vector<std::filesystem::path> Directory::getFilesPrefix(const std::filesystem::path& prefix)
{
    auto tmp = getFiles();
    auto e   = std::remove_if(tmp.begin(), tmp.end(), [&](const std::filesystem::path& s) { return !hasPrefix(s.u8string(), prefix.u8string()); });
    tmp.erase(e, tmp.end());
    return tmp;
}


std::vector<std::filesystem::path> Directory::getDirectories()
{
    std::vector<std::filesystem::path> out;

    for (auto& it : std::filesystem::directory_iterator(dirname))
    {
        if (it.is_directory())
        {
            out.push_back(std::filesystem::relative(it.path(), dirname));
        }
    }
    return out;
}


std::vector<std::filesystem::path> Directory::getDirectories(const std::filesystem::path& ending)
{
    auto tmp = getDirectories();

    auto e = std::remove_if(tmp.begin(), tmp.end(), [&](const std::filesystem::path& s) { return !hasEnding(s.u8string(), ending.u8string()); });
    tmp.erase(e, tmp.end());
    return tmp;
}

bool Directory::existsFile(const std::filesystem::path& file)
{
    auto all = getFiles();
    return std::find(all.begin(), all.end(), file) != all.end();
}

}  // namespace Saiga
