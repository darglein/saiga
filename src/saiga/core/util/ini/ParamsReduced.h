/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/Quaternion.h"
#include "saiga/core/math/Types.h"

#include <vector>
#include <filesystem>

namespace CLI
{
class App;
}

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
    virtual ~ParamsBase() {}
    std::string name_;

    // virtual void Params(Saiga::SimpleIni* ini, CLI::App* app) = 0;

    // template<class ParamIterator>
    // virtual void Params(ParamIterator* ini) = 0;
};

#define SAIGA_PARAM_STRUCT(_Name)                                          \
    using ParamStructType = _Name;                                         \
    _Name() : ParamsBase(#_Name) {}                                        \
    explicit _Name(const std::filesystem::path& file) : ParamsBase(#_Name) \
    {                                                                      \
        Load(file);                                                        \
    }

#define SAIGA_PARAM_STRUCT_FUNCTION_DEFINITIONS \
    void Load(CLI::App& app);                   \
    virtual void Load(const std::filesystem::path& file); \
    virtual void Save(const std::filesystem::path& file); \
    virtual void Print(std::ostream& strm, int column_width = 30);


#define SAIGA_PARAM_DEFAULT(_variable) (ParamStructType()._variable)

#define SAIGA_PARAM(_variable) \
    if (it) it->SaigaParam(name_, _variable, SAIGA_PARAM_DEFAULT(_variable), #_variable, "")

#define SAIGA_PARAM_COMMENT(_variable, _comment) \
    if (it) it->SaigaParam(name_, _variable, SAIGA_PARAM_DEFAULT(_variable), #_variable, _comment)

#define SAIGA_PARAM_LIST(_variable, _sep) \
    if (it) it->SaigaParamList(name_, _variable, SAIGA_PARAM_DEFAULT(_variable), #_variable, _sep, "")

// a variation where the list is also passed to the command line parser
#define SAIGA_PARAM_LIST2(_variable, _sep) \
    if (it) it->SaigaParamList(name_, _variable, SAIGA_PARAM_DEFAULT(_variable), #_variable, _sep, "")

#define SAIGA_PARAM_LIST_COMMENT(_variable, _sep, _comment) \
    if (it) it->SaigaParamList(name_, _variable, _variable, #_variable, _sep, _comment)
