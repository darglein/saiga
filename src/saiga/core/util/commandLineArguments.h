/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

#include <string>
#include <vector>
namespace Saiga
{
class SAIGA_CORE_API CommandLineArguments
{
   public:
    struct CLA
    {
        std::string longName;
        char shortName;
        std::string description;
        std::string defaultValue;
        // if this argument is an on/off flag
        bool isFlag     = false;
        bool isRequired = true;



        CLA(const std::string& longName, char shortName, const std::string& description,
            const std::string& defaultValue = "", bool isFlag = true, bool isRequired = true)
            : longName(longName),
              shortName(shortName),
              description(description),
              defaultValue(defaultValue),
              isFlag(isFlag),
              isRequired(isRequired),
              value(defaultValue)
        {
            if (!value.empty())
            {
                hasValue = true;
            }
        }

        bool valid() { return !isRequired || hasValue; }

        bool flagValue    = false;
        std::string value = "";
        bool hasValue     = false;
    };

    std::vector<CLA> arguments;


    std::string exePath;
    std::vector<std::string> args;


    std::string get(const std::string& name);
    long getLong(const std::string& name);
    bool getFlag(const std::string& name);


    template <typename... Args>
    CommandLineArguments(Args&&... args) : arguments(std::forward<Args>(args)...)
    {
    }


    void parse(int argc, char* argv[]);
    void printHelp();
};

}  // namespace Saiga
