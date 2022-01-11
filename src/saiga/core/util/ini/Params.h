/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once


#include "saiga/core/util/commandLineArguments.h"

#include "ini.h"



// The saiga param macros (below) can be used to define simple param structs in ini files.
// An example struct should look like this:
//
//  SAIGA_PARAM_STRUCT(NetworkParams)
//  {
//      SAIGA_PARAM_STRUCT_FUNCTIONS(NetworkParams);
//
//      double d        = 2;
//      long n          = 10;
//      std::string str = "blabla";
//
//      void Params()
//      {
//          SAIGA_PARAM_DOUBLE(d);
//          SAIGA_PARAM_LONG(n);
//          SAIGA_PARAM_STRING(str);
//      }
//  };
//
// To update the parameters by the command line use:
//
//      // First load from config file
//      MyParams params("config.ini");
//
//      // No update from command line
//       CLI::App app{"Example programm", "exmaple_programm"};
//      params.Load(app);
//      CLI11_PARSE(app, argc, argv);

struct ParamsBase
{
    ParamsBase(const std::string name) : name_(name) {}
    std::string name_;

    virtual void Params(Saiga::SimpleIni* ini, CLI::App* app) = 0;

    void Load(CLI::App& app) { Params(nullptr, &app); }

    virtual void Load(std::string file)
    {
        Saiga::SimpleIni ini_;
        ini_.LoadFile(file.c_str());
        Params(&ini_, nullptr);
        if (ini_.changed()) ini_.SaveFile(file.c_str());
    }

    virtual void Save(std::string file)
    {
        Saiga::SimpleIni ini_;
        ini_.LoadFile(file.c_str());
        Params(&ini_, nullptr);
        ini_.SaveFile(file.c_str());
    }
};

#define SAIGA_PARAM_STRUCT_FUNCTIONS(_Name) \
    _Name() : ParamsBase(#_Name) {}         \
    _Name(const std::string file) : ParamsBase(#_Name) { Load(file); }


#define SAIGA_PARAM(_variable)                           \
    if (ini) INI_GETADD(*ini, name_.c_str(), _variable); \
    if (app) app->add_option("--" #_variable, _variable, "", true)

#define SAIGA_PARAM_COMMENT(_variable, _comment)                           \
    if (ini) INI_GETADD_COMMENT(*ini, name_.c_str(), _variable, _comment); \
    if (app) app->add_option("--" #_variable, _variable, _comment, true)

#define SAIGA_PARAM_LIST(_variable, _sep) \
    if (ini) INI_GETADD_LIST_COMMENT(*ini, name_.c_str(), _variable, _sep, "");

// a variation where the list is also passed to the command line parser
#define SAIGA_PARAM_LIST2(_variable, _sep)                                      \
    if (ini) INI_GETADD_LIST_COMMENT(*ini, name_.c_str(), _variable, _sep, ""); \
    if (app && _sep == ' ') app->add_option("--" #_variable, _variable, "", true)

#define SAIGA_PARAM_LIST_COMMENT(_variable, _sep, _comment) \
    if (ini) INI_GETADD_LIST_COMMENT(*ini, name_.c_str(), _variable, _sep, _comment);
