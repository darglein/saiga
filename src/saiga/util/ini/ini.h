/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once


#include "saiga/util/ini/SimpleIni.h"


/**
 *  ============== Usage Example ===============
 * Note:    The "getAdd" functions add the keys if they don't exist yet.
 *          This is usefull to auto-generate a config file if it so present.
 *
 *
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
 *   cout << name << " " << w << "x" << h << " " << b << endl;
 *
 */
