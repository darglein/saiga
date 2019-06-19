/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once


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
 *   b    = ini.GetAddBool   ("window","fullscreen",false);
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
#define INI_GETADD_BOOL(_ini, _section, _variable) (_variable) = (_ini).GetAddBool(_section, #_variable, _variable)
#define INI_GETADD_LONG(_ini, _section, _variable) (_variable) = (_ini).GetAddLong(_section, #_variable, _variable)
#define INI_GETADD_STRING(_ini, _section, _variable) \
    (_variable) = (_ini).GetAddString(_section, #_variable, _variable.c_str())
#define INI_GETADD_DOUBLE(_ini, _section, _variable) (_variable) = (_ini).GetAddDouble(_section, #_variable, _variable)
