/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

#include <vector>

namespace Saiga
{
class SAIGA_GLOBAL FileChecker
{
   public:
    FileChecker();

    // searches for 'file' at all search pathes and returns the full name
    std::string getFile(const std::string& file);
    std::string operator()(const std::string& file) { return getFile(file); }



    // returns the full file name of 'file' that is relative addressed to 'basefile'
    std::string getRelative(const std::string& baseFile, const std::string& file);

    /**
     * returns the parent directory of 'file'
     * Example:
     * test/image.png
     * ->   test/
     */
    std::string getParentDirectory(const std::string& file);

    /**
     * returns the raw file name of 'file'
     * Example:
     * test/image.png
     * ->   image.png
     */
    std::string getFileName(const std::string& file);

    /**
     * Like above, but only if the file ends on "ending"
     */
    void getFiles(std::vector<std::string>& out, const std::string& predir, const std::string& ending);

    void addSearchPath(const std::string& path);
    void addSearchPath(const std::vector<std::string>& paths);

    bool existsFile(const std::string& file);


    SAIGA_GLOBAL friend std::ostream& operator<<(std::ostream& os, const FileChecker& fc);

   private:
    // all file search functions search at these pathes.
    // the first match will return.
    // the empty path is added by default.
    std::vector<std::string> searchPathes;
};


namespace SearchPathes
{
/**
 * Global search pathes used by different saiga modules.
 * These are set in framework.cpp.
 * Additional search pathes can be set in the config.ini
 */
extern SAIGA_GLOBAL FileChecker shader;
extern SAIGA_GLOBAL FileChecker image;
extern SAIGA_GLOBAL FileChecker model;
extern SAIGA_GLOBAL FileChecker font;
extern SAIGA_GLOBAL FileChecker data;
}  // namespace SearchPathes

}  // namespace Saiga
