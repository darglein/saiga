/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/util/tostring.h"
#include "saiga/util/assert.h"

namespace Saiga {

std::vector<std::string> split(const std::string &s, char delim)
{
    std::stringstream ss(s);
    std::string item;
    std::vector<std::string> elems;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

std::string leadingZeroString(int number, int characterCount)
{
    std::string n = Saiga::to_string(number);
    SAIGA_ASSERT((int)n.size() <= characterCount);
    std::string nleading(characterCount - n.size(),'0');
    nleading.insert(nleading.end(),n.begin(),n.end());
    return nleading;
}

}
