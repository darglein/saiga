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
class SAIGA_CORE_API FileChecker
{
   public:
    FileChecker();

    // searches for 'file' at all search pathes and returns the full name
    std::filesystem::path getFile(const std::filesystem::path& file);
    std::filesystem::path operator()(const std::filesystem::path& file) { return getFile(file); }

    // returns the full file name of 'file' that is relative addressed to 'basefile'
    std::filesystem::path getRelative(const std::filesystem::path& baseFile, const std::filesystem::path& file);

    /**
     * returns the parent directory of 'file'
     * Example:
     * test/image.png
     * ->   test/
     */
    static std::filesystem::path getParentDirectory(const std::filesystem::path& file);

    /**
     * returns the raw file name of 'file'
     * Example:
     * test/image.png
     * ->   image.png
     */
    std::filesystem::path getFileName(const std::filesystem::path& file);

    /**
     * Appends the full names of all regular files in the directory 'predir' whose names end on 'ending'.
     * 'predir' is searched at all search pathes; a directory reached through several search pathes is listed once.
     * The files of each directory are sorted by name.
     * Example:
     * getFiles(out, "textures", ".png")
     * ->   out = {"<search path>/textures/a.png", "<search path>/textures/b.png", ...}
     */
    void getFiles(std::vector<std::string>& out, const std::string& predir, const std::string& ending);

    bool existsFile(const std::filesystem::path& file);

    void addSearchPath(const std::filesystem::path& path);
    void addSearchPath(const std::vector<std::filesystem::path>& paths);

    /**
     * If disabled, only absolute candidates are returned: relative file names and relative search pathes (including
     * the default "" and ".") are not resolved against the working directory.
     * Applications that load trusted data (shaders, fonts, DLL folders, ...) should disable this, because the working
     * directory is often a user-chosen folder, e.g. the folder of an opened document.
     */
    void setSearchWorkingDirectory(bool enabled) { searchWorkingDirectory = enabled; }

    SAIGA_CORE_API friend std::ostream& operator<<(std::ostream& os, const FileChecker& fc);

   private:
    bool isAllowed(const std::filesystem::path& file) const;

    // all file search functions search at these pathes.
    // the first match will return.
    // the empty path is added by default.
    std::vector<std::filesystem::path> searchPathes;
    bool searchWorkingDirectory = true;
};


namespace SearchPathes
{
/**
 * Global search pathes used by different saiga modules.
 * These are set in framework.cpp.
 * Additional search pathes can be set in the config.ini
 */
SAIGA_CORE_API extern FileChecker shader;
SAIGA_CORE_API extern FileChecker image;
SAIGA_CORE_API extern FileChecker model;
SAIGA_CORE_API extern FileChecker font;
SAIGA_CORE_API extern FileChecker data;
}  // namespace SearchPathes

}  // namespace Saiga
