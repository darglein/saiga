#pragma once

#include <saiga/config.h>
#include <string>
#include <iostream>

class SAIGA_GLOBAL FileChecker{
public:

    //returns the full file name of 'file' that is relative addressed to 'basefile'
    std::string getRelative(const std::string& baseFile, const std::string& file);

    /**
     * returns the parent directory of 'file'
     * Example:
     * test/image.png
     * ->   test/
     */
    std::string getParentDirectory(const std::string& file);
};
