/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "saiga/core/util/fileChecker.h"

#include "saiga/core/util/FileSystem.h"
#include "saiga/core/util/directory.h"

#include "internal/noGraphicsAPI.h"

#include <fstream>
#include <iostream>

namespace Saiga
{
FileChecker::FileChecker()
{
    searchPathes.push_back("");
    searchPathes.push_back(".");
}

std::filesystem::path FileChecker::getFile(const std::filesystem::path& file)
{
    if (file.empty()) return {};

    // Check without search pathes
    if (existsFile(file)) return file;

    for (const std::filesystem::path& path : searchPathes)
    {
        // Do not generate a double '/'
        std::filesystem::path fullName = path / file;
        if (existsFile(fullName))
        {
            return fullName;
        }
    }
    return {};
}

void FileChecker::addSearchPath(const std::filesystem::path& path)
{
    searchPathes.push_back(path);
}

void FileChecker::addSearchPath(const std::vector<std::filesystem::path>& paths)
{
    for (auto& s : paths) addSearchPath(s);
}

std::filesystem::path FileChecker::getRelative(const std::filesystem::path& baseFile, const std::filesystem::path& file)
{
    // first check at path relative to the parent
    auto parent = getParentDirectory(baseFile);
    std::filesystem::path relativeName = parent / file;
    if (existsFile(relativeName))
    {
        return relativeName;
    }
    return getFile(file);
}

std::filesystem::path FileChecker::getParentDirectory(const std::filesystem::path& file)
{
    return file.parent_path();
}

std::filesystem::path FileChecker::getFileName(const std::filesystem::path& file)
{
    return file.filename();
}

bool FileChecker::existsFile(const std::filesystem::path& file)
{
    try
    {
        return std::filesystem::exists(file);
    }
    catch (std::filesystem::filesystem_error& e)
    {
        // A filesystem error when checking if a file exists should usually not happen.
        // But if it does this file is certainly not readable.
        return false;
    }
}

std::ostream& operator<<(std::ostream& os, const FileChecker& fc)
{
    os << "File Checker - Search Pathes:" << std::endl;
    for (auto s : fc.searchPathes)
    {
        os << "   '" << s << "'" << std::endl;
    }
    return os;
}

namespace SearchPathes
{
FileChecker shader;
FileChecker image;
FileChecker model;
FileChecker font;
FileChecker data;
}  // namespace SearchPathes
}  // namespace Saiga
