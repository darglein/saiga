/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

#include <string>
#include <vector>
#ifdef _WIN32
#    include "saiga/core/util/windows_dirent.h"
#else
#    include <dirent.h>
#endif
namespace Saiga
{
class SAIGA_CORE_API Directory
{
   public:
    std::string dirname;
    DIR* dir = nullptr;
    Directory(const std::string& dir);
    ~Directory();


    /**
     * Gets all regular files in this directory.
     */
    void getFiles(std::vector<std::string>& out);

    /**
     * Like above, but only if the file ends on "ending"
     */
    void getFiles(std::vector<std::string>& out, const std::string& ending);

    /**
     * Like above, but only if the file starts with "prefix"
     */
    void getFilesPrefix(std::vector<std::string>& out, const std::string& prefix);

    /**
     * Gets all directories in this directory.
     */
    void getDirectories(std::vector<std::string>& out);
    void getDirectories(std::vector<std::string>& out, const std::string& ending);


    bool existsFile(const std::string& file);

    std::string operator()() { return dirname; }
};

}  // namespace Saiga
