/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/util/commandLineArguments.h"
#include "saiga/util/assert.h"

namespace Saiga {

std::string CommandLineArguments::get(std::string name)
{
    for(auto& arg : arguments)
    {
        if(arg.long_name == name)
            return arg.value;
    }
    SAIGA_ASSERT(0);
    return "";
}

void CommandLineArguments::parse(int argc, char *argv[])
{
    SAIGA_ASSERT(argc >= 1);


    cout << "parsing " << argc << " commandline arguments... " << endl;

    exePath = argv[0];

    for(int i = 1; i < argc; ++i)
    {
        args.push_back(argv[i]);
    }


    //    for(auto s : args)
    //        cout << s << endl;

    std::string minus = "-";
    std::string minusminus = "--";

    int currentArg = 0;

    for(;currentArg < args.size(); ++currentArg)
    {
        auto str = args[currentArg];

        bool longName;



        //check if it starts with "-" or "--"
        if(!str.compare(0,minusminus.size(),minusminus))
        {
            longName = true;

        }else if(!str.compare(0,minus.size(),minus))
        {
            longName = false;
        }else
        {
            cout << "invalid parameter" << endl;
            SAIGA_ASSERT(0);
            return;
        }

        auto subStr = longName ? str.substr(2) : str.substr(1);

        //        cout << "substr " << subStr << endl;

        //split into name and value
        auto equalPos = subStr.find('=');

        bool isFlag = equalPos == std::string::npos;


        auto name = subStr.substr(0,equalPos);
        auto value = isFlag ? "1" : subStr.substr(equalPos+1);

        if( (longName && "help"==name) ||
                (!longName && 'h'==name[0]) )
        {
            printHelp();
            continue;
        }

        //find corresponding
        for(auto& arg : arguments)
        {
            if( (longName && arg.long_name==name) ||
                    (!longName && arg.short_name==name[0]) )
            {
                arg.value =value;
            }
        }

    }
}

void CommandLineArguments::printHelp()
{
    cout << "CommandLineArguments Help" << endl;
    for(auto& arg : arguments)
    {
        if(arg.long_name != "")
        {
            cout << "[--" << arg.long_name << "]";
        }
        if(arg.short_name != 0)
        {
            cout << "[-" << arg.short_name << "]";
        }

        cout << " " << arg.description;

        if(arg.flag)
        {
        }else
        {

            cout << " default=[" << arg.value << "]";
        }
        cout << endl;
    }
}

}
