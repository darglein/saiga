/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once


#include "saiga/core/math/math.h"
#include "saiga/core/util/tostring.h"

#include "SimpleIni.h"

/**
 *  ============== Usage Example ===============
 * Note:    The "getAdd" functions add the keys if they don't exist yet.
 *          This is usefull to auto-generate a config file if it so present.
 *
 *   // Checkout the full source at samples/ini/main.cpp
 *   std::string name;
 *   int w,h;
 *   bool b;
 *
 *   Saiga::SimpleIni ini;
 *   ini.LoadFile("config.ini");
 *
 *   name = ini.GetAddString ("window","name","Test Window");
 *   w    = ini.GetAddLong   ("window","width",1280);
 *   h    = ini.GetAddDouble ("window","height",720);
 *   b    = ini.GetAddBool   ("window","Fullscreen",false);
 *
 *   if(ini.changed()) ini.SaveFile("config.ini");
 *
 *   std::cout << name << " " << w << "x" << h << " " << b << std::endl;
 *
 */



/**
 * Helper macros for creating the most common use-case:
 *
 * - The variable x is loaded from the ini file
 * - The default value is the previous value of x
 * - The name in the .ini file is the actual variable name
 *
 * Example usage:
 *
 * double foo = 3.14;
 * INI_GETADD_DOUBLE(ini, "Math", foo);
 */
#define INI_GETADD_BOOL_COMMENT(_ini, _section, _variable, _comment) \
    (_variable) = (_ini).GetAddBool(_section, #_variable, _variable, _comment)
#define INI_GETADD_LONG_COMMENT(_ini, _section, _variable, _comment) \
    (_variable) = (_ini).GetAddLong(_section, #_variable, _variable, _comment)
#define INI_GETADD_STRING_COMMENT(_ini, _section, _variable, _comment) \
    (_variable) = (_ini).GetAddString(_section, #_variable, _variable.c_str(), _comment)
#define INI_GETADD_DOUBLE_COMMENT(_ini, _section, _variable, _comment) \
    (_variable) = (_ini).GetAddDouble(_section, #_variable, _variable, _comment)
#define INI_GETADD_STRING_LIST_COMMENT(_ini, _section, _variable, _sep, _comment) \
    (_variable) = Saiga::split(                                                   \
        (_ini).GetAddString(_section, #_variable, Saiga::concat(_variable, _sep).c_str(), _comment), _sep)
#define INI_GETADD_MATRIX_COMMENT(_ini, _section, _variable, _comment) \
    StringToMatrix((_ini).GetAddString(_section, #_variable, MatrixToString(_variable).c_str(), _comment), _variable)

#define INI_GETADD_BOOL(_ini, _section, _variable) INI_GETADD_BOOL_COMMENT(_ini, _section, _variable, 0)
#define INI_GETADD_LONG(_ini, _section, _variable) INI_GETADD_LONG_COMMENT(_ini, _section, _variable, 0)
#define INI_GETADD_STRING(_ini, _section, _variable) INI_GETADD_STRING_COMMENT(_ini, _section, _variable, 0)
#define INI_GETADD_DOUBLE(_ini, _section, _variable) INI_GETADD_DOUBLE_COMMENT(_ini, _section, _variable, 0)



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
struct ParamsBase
{
    ParamsBase(const std::string name) : name_(name) {}
    std::string name_;

    virtual void Params(Saiga::SimpleIni& ini_) = 0;

    virtual void Load(std::string file)
    {
        Saiga::SimpleIni ini_;
        ini_.LoadFile(file.c_str());
        Params(ini_);
        if (ini_.changed()) ini_.SaveFile(file.c_str());
    }

    virtual void Save(std::string file)
    {
        Saiga::SimpleIni ini_;
        ini_.LoadFile(file.c_str());
        Params(ini_);
        ini_.SaveFile(file.c_str());
    }
};

#define SAIGA_PARAM_STRUCT_FUNCTIONS(_Name) \
    _Name() : ParamsBase(#_Name) {}         \
    _Name(const std::string file) : ParamsBase(#_Name) { Load(file); }


#define SAIGA_PARAM_BOOL(_variable) INI_GETADD_BOOL(ini_, name_.c_str(), _variable)
#define SAIGA_PARAM_LONG(_variable) INI_GETADD_LONG(ini_, name_.c_str(), _variable)
#define SAIGA_PARAM_STRING(_variable) INI_GETADD_STRING(ini_, name_.c_str(), _variable)
#define SAIGA_PARAM_STRING_LIST(_variable, _sep) INI_GETADD_STRING_LIST_COMMENT(ini_, name_.c_str(), _variable, _sep, 0)
#define SAIGA_PARAM_STRING_LIST_COMMENT(_variable, _sep, _comment) \
    INI_GETADD_STRING_LIST_COMMENT(ini_, name_.c_str(), _variable, _sep, _comment)
#define SAIGA_PARAM_DOUBLE(_variable) INI_GETADD_DOUBLE(ini_, name_.c_str(), _variable)

#define SAIGA_PARAM_STRING_COMMENT(_variable, _comment) \
    INI_GETADD_STRING_COMMENT(ini_, name_.c_str(), _variable, _comment)
