/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "file.h"

#include <fstream>

namespace Saiga {
namespace File {

std::vector<unsigned char> loadFileBinary(const std::string &file)
{
    std::vector<unsigned char> result;

    std::ifstream is(file, std::ios::binary | std::ios::in | std::ios::ate);
    if (!is.is_open()) {
        cout << "File not found " << file << endl;
        return result;
    }

    size_t size = is.tellg();
    result.resize(size);
    is.seekg(0, std::ios::beg);
    is.read( (char*)result.data(), size);
    is.close();
    return result;

}

std::string loadFileString(const std::string &file)
{
    std::string fileContent;
    std::ifstream fileStream(file, std::ios::in);
    if (!fileStream.is_open()) {
        cout << "File not found " << file << endl;
        return "";
    }
    std::string line = "";
    while (!fileStream.eof()) {
        getline(fileStream, line);
        fileContent.append(line + "\n");
    }
    fileStream.close();
    return fileContent;
}


}

}
