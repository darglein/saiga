/**
 * Copyright (c) 2021 Darius Rückert
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
    Directory(const std::string& dir);
    ~Directory();


    /**
     * Gets all regular files in this directory.
     */
    std::vector<std::string> getFiles();

    /**
     * Like above, but only if the file ends on "ending"
     */
    std::vector<std::string> getFilesEnding(const std::string& ending);

    /**
     * Like above, but only if the file starts with "prefix"
     */
    std::vector<std::string> getFilesPrefix(const std::string& prefix);

    /**
     * Gets all directories in this directory.
     */
    std::vector<std::string> getDirectories();
    std::vector<std::string> getDirectories(const std::string& ending);


    bool existsFile(const std::string& file);

    std::string operator()() { return dirname; }

   private:
    std::string dirname;
    DIR* dir = nullptr;
};

}  // namespace Saiga
