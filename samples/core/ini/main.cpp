/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/util/ini/ini.h"
#include "saiga/util/math.h"
#include "saiga/util/tostring.h"

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
    mat4 m = identityMat4();
    m[1][0] = 1;  // row 0 and col 1

    name = ini.GetAddString("window", "name", "Test Window");
    w    = ini.GetAddLong("window", "width", 1280);
    h    = ini.GetAddDouble("window", "height", 720);
    b    = ini.GetAddBool("window", "fullscreen", false);
    m    = Saiga::mat4FromString(ini.GetAddString("window", "viewmatrix", Saiga::to_string(m).c_str()));

    cout << name << " " << w << "x" << h << " " << b << " " << endl << m << endl;

    if (ini.changed()) ini.SaveFile(fileName);
}
