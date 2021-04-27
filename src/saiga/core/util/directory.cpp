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
Directory::Directory(const std::string& dir)
{
    dirname = dir;
    if ((this->dir = opendir(dir.c_str())) == NULL)
    {
        //        std::cout<<"could not open directory: "<<dir<<std::endl;
        //        SAIGA_ASSERT(0);
    }
}

Directory::~Directory()
{
    if (dir) closedir(dir);
}

std::vector<std::string> Directory::getFiles()
{
    std::vector<std::string> out;
    if (!dir) return out;

    struct dirent* ent;
    rewinddir(dir);
    while ((ent = readdir(dir)) != NULL)
    {
        if (ent->d_type == DT_REG)
        {
            std::string str(ent->d_name);
            out.push_back(str);
        }
        else if (ent->d_type == DT_UNKNOWN)
        {
            // On some filesystems like XFS the d_type is always DT_UNKNOWN.
            // We need to use stat to check if it's a regular file. (Thanks to Samuel Nelson)
            std::string fullFileName = dirname + "/" + std::string(ent->d_name);
            struct stat st;
            int ret = stat(fullFileName.c_str(), &st);
            SAIGA_ASSERT(ret == 0);
            if (S_ISREG(st.st_mode))
            {
                std::string str(ent->d_name);
                out.push_back(str);
            }
        }
    }
    return out;
}


std::vector<std::string> Directory::getFilesEnding(const std::string& ending)
{
    auto tmp = getFiles();
    auto e   = std::remove_if(tmp.begin(), tmp.end(), [&](std::string& s) { return !hasEnding(s, ending); });
    tmp.erase(e, tmp.end());
    return tmp;
}

std::vector<std::string> Directory::getFilesPrefix(const std::string& prefix)
{
    auto tmp = getFiles();
    auto e   = std::remove_if(tmp.begin(), tmp.end(), [&](std::string& s) { return !hasPrefix(s, prefix); });
    tmp.erase(e, tmp.end());
    return tmp;
}


std::vector<std::string> Directory::getDirectories()
{
    std::vector<std::string> out;
    if (!dir) return out;

    struct dirent* ent;
    while ((ent = readdir(dir)) != NULL)
    {
        if (ent->d_type == DT_DIR)
        {
            std::string str(ent->d_name);
            out.push_back(str);
        }
    }
    return out;
}


std::vector<std::string> Directory::getDirectories(const std::string& ending)
{
    auto tmp = getDirectories();

    auto e = std::remove_if(tmp.begin(), tmp.end(), [&](std::string& s) { return !hasEnding(s, ending); });
    tmp.erase(e, tmp.end());
    return tmp;
    //    for(auto& str : tmp)
    //    {
    //        if(hasEnding(str,ending))
    //        {
    //            out.push_back(str);
    //        }
    //    }
}

bool Directory::existsFile(const std::string& file)
{
    auto all = getFiles();
    return std::find(all.begin(), all.end(), file) != all.end();
}

}  // namespace Saiga
