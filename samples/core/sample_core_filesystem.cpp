/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"
#include "saiga/core/Core.h"

#include "saiga/core/util/FileSystem.h"
#include <iostream>

namespace fs = std::filesystem;
using namespace std;

int main()
{
    fs::path aPath{"./path/to/file.txt"};

    std::cout << "Parent path: " << aPath.parent_path() << std::endl;
    std::cout << "Filename: " << aPath.filename() << std::endl;
    std::cout << "Extension: " << aPath.extension() << std::endl;

    return 0;
}
