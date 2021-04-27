/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/util/tostring.h"

#include "saiga/core/util/assert.h"

#include "internal/noGraphicsAPI.h"

#include <array>

namespace Saiga
{
std::vector<std::string> split(const std::string& s, char delim)
{
    std::stringstream ss(s);
    std::string item;
    std::vector<std::string> elems;
    while (std::getline(ss, item, delim))
    {
        elems.push_back(item);
    }
    return elems;
}

std::string concat(const std::vector<std::string>& s, char delim)
{
    std::string res;
    for (unsigned int i = 0; i < s.size(); ++i)
    {
        if (i > 0) res += delim;
        res += s[i];
    }
    return res;
}

std::string leadingZeroString(int number, int characterCount)
{
    std::string n = Saiga::to_string(number);
    if (n.size() > characterCount)
    {
        return n;
    }

    std::string nleading(characterCount - n.size(), '0');
    nleading.insert(nleading.end(), n.begin(), n.end());
    return nleading;
}

bool hasEnding(std::string const& fullString, std::string const& ending)
{
    if (fullString.length() >= ending.length())
    {
        return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
    }
    else
    {
        return false;
    }
}

bool hasPrefix(const std::string& fullString, const std::string& prefix)
{
    if (fullString.length() >= prefix.length())
    {
        return (0 == fullString.compare(0, prefix.length(), prefix));
    }
    else
    {
        return false;
    }
}

std::string fileEnding(const std::string& str)
{
    // search last '.' from the end
    for (auto it = str.rbegin(); it != str.rend(); ++it)
    {
        if (*it == '.')
        {
            auto d = std::distance(it, str.rend());
            return str.substr(d, str.size());
        }
    }
    return std::string();
}

std::string removeFileEnding(const std::string& str)
{
    // search last '.' from the end
    for (auto it = str.rbegin(); it != str.rend(); ++it)
    {
        if (*it == '.')
        {
            auto d = std::distance(it, str.rend());
            return str.substr(0, d - 1);
        }
    }
    return std::string();
}


std::string sizeToString(size_t size, size_t base, size_t max, const char* sep, std::streamsize precision)
{
    const static std::array<const char*, 6> prefixes = {"B", "kB", "MB", "GB", "TB", "PB"};

    double float_size = size;

    int count = 0;

    while (float_size > max)
    {
        ++count;
        float_size /= base;
    }

    std::stringstream size_stream;
    size_stream.setf(std::ios::fixed, std::ios::floatfield);
    size_stream.precision(count > 0 ? precision : 0);
    size_stream << float_size << sep << prefixes[count];

    return size_stream.str();
}



}  // namespace Saiga
