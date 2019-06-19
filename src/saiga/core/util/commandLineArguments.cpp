/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/util/commandLineArguments.h"

#include "saiga/core/util/assert.h"
#include "saiga/core/util/tostring.h"

#include "internal/noGraphicsAPI.h"

#include <algorithm>
#include <iostream>
namespace Saiga
{
std::string CommandLineArguments::get(const std::string& name)
{
    for (auto& arg : arguments)
    {
        if (arg.longName == name)
        {
            SAIGA_ASSERT(arg.valid());
            return arg.value;
        }
    }
    SAIGA_ASSERT(0);
    return "";
}

long CommandLineArguments::getLong(const std::string& name)
{
    return Saiga::to_long(get(name));
}

bool CommandLineArguments::getFlag(const std::string& name)
{
    std::string value = get(name);
    std::transform(value.begin(), value.end(), value.begin(), ::tolower);
    return (value == "1" || value == "on" || value == "true" || value == "yes");
}

void CommandLineArguments::parse(int argc, char* argv[])
{
    SAIGA_ASSERT(argc >= 1);
    exePath = argv[0];

    for (int i = 1; i < argc; ++i)
    {
        args.push_back(argv[i]);
    }


    //    for(auto s : args)
    //        std::cout << s << std::endl;

    std::string minus      = "-";
    std::string minusminus = "--";

    int currentArg = 0;

    for (; currentArg < (int)args.size(); ++currentArg)
    {
        auto str = args[currentArg];

        bool longName;



        // check if it starts with "-" or "--"
        if (!str.compare(0, minusminus.size(), minusminus))
        {
            longName = true;
        }
        else if (!str.compare(0, minus.size(), minus))
        {
            longName = false;
        }
        else
        {
            std::cout << "invalid parameter" << std::endl;
            SAIGA_ASSERT(0);
            return;
        }

        auto subStr = longName ? str.substr(2) : str.substr(1);

        //        std::cout << "substr " << subStr << std::endl;

        // split into name and value
        auto equalPos = subStr.find('=');

        bool isFlag = equalPos == std::string::npos;


        auto name  = subStr.substr(0, equalPos);
        auto value = isFlag ? "1" : subStr.substr(equalPos + 1);

        if ((longName && "help" == name) || (!longName && 'h' == name[0]))
        {
            printHelp();
            continue;
        }

        // find corresponding
        for (auto& arg : arguments)
        {
            if ((longName && arg.longName == name) || (!longName && arg.shortName == name[0]))
            {
                arg.value = value;
            }
        }
    }
}

void CommandLineArguments::printHelp()
{
    SAIGA_ASSERT(!arguments.empty());
    for (auto& arg : arguments)
    {
        if (arg.longName != "")
        {
            std::cout << "--" << arg.longName;
        }
        if (arg.shortName != 0)
        {
            std::cout << "  -" << arg.shortName;
        }
        std::cout << std::endl;

        std::cout << "    Description: " << arg.description << std::endl;


        {
            std::cout << "    Default: " << arg.defaultValue << std::endl;
        }

        if (arg.isFlag)
        {
            std::cout << "    Flag: yes" << std::endl;
        }
        else
        {
            std::cout << "    Flag: no" << std::endl;
        }

        if (arg.isRequired)
        {
            std::cout << "    Required: yes" << std::endl;
        }
        else
        {
            std::cout << "    Required: no" << std::endl;
        }


        std::cout << std::endl;
    }
    exit(0);
}

}  // namespace Saiga
