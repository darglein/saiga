/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"
#include "saiga/core/Core.h"

#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;
using namespace std;

int main()
{
    fs::path aPath{"./path/to/file.txt"};

    cout << "Parent path: " << aPath.parent_path() << endl;
    cout << "Filename: " << aPath.filename() << endl;
    cout << "Extension: " << aPath.extension() << endl;

    return 0;
}
