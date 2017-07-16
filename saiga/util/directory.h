#pragma once

#include "saiga/config.h"

#include <vector>
#ifdef _WIN32
#include "util/windows_dirent.h"
#else
#include <dirent.h>
#endif

namespace Saiga {

class SAIGA_GLOBAL Directory {
public:
    std::string dirname;
    DIR *dir;
    Directory(const std::string& dir);
    ~Directory();

    void getFiles(std::vector<std::string> &out);
    void getFiles(std::vector<std::string> &out, const std::string &ending);
};

}
