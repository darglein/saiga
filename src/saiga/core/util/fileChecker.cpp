/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "saiga/core/util/fileChecker.h"

#include "saiga/core/util/FileSystem.h"
#include "saiga/core/util/directory.h"
#include "saiga/core/util/string_to_path.h"

#include "internal/noGraphicsAPI.h"

#include <algorithm>
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
    if (isAllowed(file) && existsFile(file)) return file;

    for (const std::filesystem::path& path : searchPathes)
    {
        // Do not generate a double '/'
        std::filesystem::path fullName = path / file;
        if (isAllowed(fullName) && existsFile(fullName))
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
    if (isAllowed(relativeName) && existsFile(relativeName))
    {
        return relativeName;
    }
    return getFile(file);
}

void FileChecker::getFiles(std::vector<std::string>& out, const std::string& predir, const std::string& ending)
{
    std::vector<std::filesystem::path> visited;
    for (const std::filesystem::path& path : searchPathes)
    {
        std::filesystem::path dir = path / string_to_path(predir);
        std::error_code ec;
        if (!isAllowed(dir) || !std::filesystem::is_directory(dir, ec)) continue;

        // The default search pathes "" and "." both point to the working directory.
        auto canonical = std::filesystem::weakly_canonical(dir, ec);
        if (ec) continue;
        if (std::find(visited.begin(), visited.end(), canonical) != visited.end()) continue;
        visited.push_back(canonical);

        std::vector<std::filesystem::path> files;
        try
        {
            files = Directory(dir).getFilesEnding(string_to_path(ending));
        }
        catch (std::filesystem::filesystem_error&)
        {
            // Not readable, e.g. missing permissions.
            continue;
        }
        std::sort(files.begin(), files.end());
        for (auto& f : files)
        {
            out.push_back(path_to_string(dir / f));
        }
    }
}

std::filesystem::path FileChecker::getParentDirectory(const std::filesystem::path& file)
{
    return file.parent_path();
}

std::filesystem::path FileChecker::getFileName(const std::filesystem::path& file)
{
    return file.filename();
}

bool FileChecker::isAllowed(const std::filesystem::path& file) const
{
    return searchWorkingDirectory || file.is_absolute();
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
