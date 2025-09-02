/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

#include <string>
#include <vector>
#include <filesystem>

namespace Saiga
{
class SAIGA_CORE_API Directory
{
   public:
    Directory(const std::filesystem::path& dir);
    ~Directory();


    /**
     * Gets all regular files in this directory.
     */
    std::vector<std::filesystem::path> getFiles();

    /**
     * Like above, but only if the file ends on "ending"
     */
    std::vector<std::filesystem::path> getFilesEnding(const std::filesystem::path& ending);

    /**
     * Like above, but only if the file starts with "prefix"
     */
    std::vector<std::filesystem::path> getFilesPrefix(const std::filesystem::path& prefix);

    /**
     * Gets all directories in this directory.
     */
    std::vector<std::filesystem::path> getDirectories();
    std::vector<std::filesystem::path> getDirectories(const std::filesystem::path& ending);


    bool existsFile(const std::filesystem::path& file);

    std::filesystem::path operator()() { return dirname; }

   private:
       std::filesystem::path dirname;
};

}  // namespace Saiga
