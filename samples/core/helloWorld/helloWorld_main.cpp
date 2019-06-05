/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"
#include "saiga/core/Core.h"


using namespace Saiga;

int main()
{
    SaigaParameters sp;
    initSample(sp);
    initSaiga(sp);
    cout << ConsoleColor::RED << "Hello World! :D" << endl;
    cout << ConsoleColor::GREEN << "Hello World! :D" << ConsoleColor::RESET << endl;
    return 0;
}
