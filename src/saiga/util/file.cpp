/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "file.h"
#include "saiga/util/assert.h"

#include <fstream>
#include "internal/noGraphicsAPI.h"

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

    if(fileContent.size() >= 3)
    {
        //remove utf8 bom
        const unsigned char* data = (const unsigned char*)fileContent.data();
        if (data[0] == 0XEF && data[1] == 0XBB && data[2] == 0XBF)
        {
            fileContent.erase(0,3);
        }
    }

    return fileContent;
}

void saveFileBinary(const std::string &file, const void* data, size_t size)
{
    std::ofstream is(file, std::ios::binary | std::ios::out);
    if (!is.is_open()) {
        cout << "File not found " << file << endl;
        return;
    }

    is.write( (const char*)data,size);
    is.close();
}


}
}
