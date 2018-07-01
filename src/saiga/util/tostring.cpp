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

bool hasEnding (std::string const &fullString, std::string const &ending)
{
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

std::string fileEnding(const std::string &str)
{
    //search last '.' from the end
    for(auto it = str.rbegin() ; it != str.rend(); ++it){
        if(*it == '.')
        {
            auto d = std::distance(it,str.rend());
            return str.substr(d,str.size());
        }
    }
    return std::string();
}

std::string removeFileEnding(const std::string &str)
{
    //search last '.' from the end
    for(auto it = str.rbegin() ; it != str.rend(); ++it){
        if(*it == '.')
        {
            auto d = std::distance(it,str.rend());
            return str.substr(0,d);
        }
    }
    return std::string();
}


}
