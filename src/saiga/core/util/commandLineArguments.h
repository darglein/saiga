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
class SAIGA_CORE_API CommandLineArguments
{
   public:
    struct CLA
    {
        std::string long_name;
        char short_name;
        std::string value;
        std::string description;
        bool flag;
        bool required;
    };

    std::vector<CLA> arguments;


    std::string exePath;
    std::vector<std::string> args;


    std::string get(std::string name);

    void parse(int argc, char* argv[]);
    void printHelp();
};

}  // namespace Saiga
