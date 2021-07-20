/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/Core.h"


using namespace Saiga;

int main(int argc, char* args[])
{
    /**
     * This sample demonstrates the use of the ini class.
     */
    auto fileName = "example.ini";

    Saiga::Ini ini;
    ini.LoadFile(fileName);


    std::string name;
    int w, h;
    bool b;
    mat4 m  = mat4::Identity();
    m(0, 1) = 1;  // row 0 and col 1

    name = ini.GetAddString("window", "name", "Test Window");
    w    = ini.GetAddLong("window", "width", 1280);
    h    = ini.GetAddDouble("window", "height", 720);
    b    = ini.GetAddBool("window", "fullscreen", false);
    Saiga::fromIniString(ini.GetAddString("window", "viewmatrix", Saiga::toIniString(m).c_str()), m);

    std::cout << name << " " << w << "x" << h << " " << b << " " << std::endl << m << std::endl;

    if (ini.changed()) ini.SaveFile(fileName);
}
