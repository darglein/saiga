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

std::string FileChecker::getFile(const std::string& file)
{
    if (file.empty()) return "";

    // Check without search pathes
    if (existsFile(file)) return file;


    for (std::string& path : searchPathes)
    {
        // Do not generate a double '/'
        std::string fullName = file.front() == '/' ? path + file : path + "/" + file;
        if (existsFile(fullName))
        {
            return fullName;
        }
    }
    return "";
}

std::string FileChecker::getRelative(const std::string& baseFile, const std::string& file)
{
    // first check at path relative to the parent
    auto parent              = getParentDirectory(baseFile);
    std::string relativeName = parent + file;
    if (existsFile(relativeName))
    {
        return relativeName;
    }
    return getFile(file);
}

std::string FileChecker::getParentDirectory(const std::string& file)
{
    // search last '/' from the end
    for (auto it = file.rbegin(); it != file.rend(); ++it)
    {
        if (*it == '/')
        {
            auto d = std::distance(it, file.rend());
            return file.substr(0, d);
        }
    }
    return "";
}

std::string FileChecker::getFileName(const std::string& file)
{
    // search last '/' from the end
    for (auto it = file.rbegin(); it != file.rend(); ++it)
    {
        if (*it == '/')
        {
            auto d = std::distance(it, file.rend());
            return file.substr(d);
        }
    }
    return "";
}

void FileChecker::getFiles(std::vector<std::string>& out, const std::string& predir, const std::string& ending)
{
    for (std::string& path : searchPathes)
    {
        std::string dir = path + "/" + predir;
        std::cout << dir << std::endl;
        Directory d(dir);


        auto tmp = d.getFilesEnding(ending);
        for (auto& s : tmp)
        {
            s = dir + "/" + s;
        }
        out.insert(out.end(), tmp.begin(), tmp.end());
    }
}
void FileChecker::addSearchPath(const std::string& path)
{
    searchPathes.push_back(path);
}

void FileChecker::addSearchPath(const std::vector<std::string>& paths)
{
    for (auto& s : paths) addSearchPath(s);
}

bool FileChecker::existsFile(const std::string& file)
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
