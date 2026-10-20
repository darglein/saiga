/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/util/tostring.h"

#include "saiga/core/util/assert.h"

#include "internal/noGraphicsAPI.h"

#include <array>
#include <cctype>
#include <charconv>
#include <limits>
#include <locale>

namespace Saiga
{
namespace
{
const char* SkipWhitespace(const char* begin, const char* end)
{
    while (begin != end && std::isspace((unsigned char)*begin)) ++begin;
    return begin;
}

const char* SkipWhitespaceBack(const char* begin, const char* end)
{
    while (end != begin && std::isspace((unsigned char)end[-1])) --end;
    return end;
}

// Parses a number at the start of [begin,end). Returns the end of the number or nullptr if there is none.
template <typename T>
const char* ParseNumberPrefix(const char* begin, const char* end, T& out)
{
    // from_chars does not accept a leading '+'
    if (begin != end && *begin == '+' && begin + 1 != end && begin[1] != '-') ++begin;

#if defined(__cpp_lib_to_chars)
    constexpr bool use_from_chars = true;
#else
    constexpr bool use_from_chars = std::is_integral<T>::value;
#endif

    if constexpr (use_from_chars)
    {
        T value;
        auto result = std::from_chars(begin, end, value);
        if (result.ec != std::errc()) return nullptr;
        out = value;
        return result.ptr;
    }
    else
    {
        std::istringstream strm(std::string(begin, end));
        strm.imbue(std::locale::classic());
        T value;
        strm >> value;
        if (strm.fail()) return nullptr;
        out = value;
        return strm.eof() ? end : begin + (size_t)strm.tellg();
    }
}

template <typename T>
bool ParseNumberStrict(const std::string& str, T& out)
{
    const char* begin = SkipWhitespace(str.data(), str.data() + str.size());
    const char* end   = SkipWhitespaceBack(begin, str.data() + str.size());
    if (begin == end) return false;
    T value;
    const char* number_end = ParseNumberPrefix(begin, end, value);
    if (number_end != end) return false;
    out = value;
    return true;
}

template <typename T>
std::string ToStringRoundtrip(T v)
{
#if defined(__cpp_lib_to_chars)
    char buffer[64];
    auto result = std::to_chars(buffer, buffer + sizeof(buffer), v);
    if (result.ec == std::errc()) return std::string(buffer, result.ptr);
#endif
    std::ostringstream strm;
    strm.imbue(std::locale::classic());
    strm << std::setprecision(std::numeric_limits<T>::max_digits10) << v;
    return strm.str();
}
}  // namespace

bool ParseNumber(const std::string& str, double& out)
{
    return ParseNumberStrict(str, out);
}
bool ParseNumber(const std::string& str, float& out)
{
    return ParseNumberStrict(str, out);
}
bool ParseNumber(const std::string& str, long long& out)
{
    return ParseNumberStrict(str, out);
}
bool ParseNumber(const std::string& str, long& out)
{
    return ParseNumberStrict(str, out);
}
bool ParseNumber(const std::string& str, int& out)
{
    return ParseNumberStrict(str, out);
}

std::string NumberToStringRoundtrip(double v)
{
    return ToStringRoundtrip(v);
}
std::string NumberToStringRoundtrip(float v)
{
    return ToStringRoundtrip(v);
}

double ParseDoublePrefix(const char* str, size_t length)
{
    const char* begin = SkipWhitespace(str, str + length);
    double value      = 0;
    if (!ParseNumberPrefix(begin, str + length, value)) return 0;
    return value;
}

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
    if (characterCount < 0 || n.size() >= size_t(characterCount))
    {
        return n;
    }

    // The zeros go after the sign: -5 -> "-05"
    size_t sign = number < 0 ? 1 : 0;
    n.insert(sign, size_t(characterCount) - n.size(), '0');
    return n;
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
    // search last '.' from the end, but only in the file name
    for (auto it = str.rbegin(); it != str.rend(); ++it)
    {
        if (*it == '/' || *it == '\\') break;
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
    const static std::array<const char*, 7> prefixes = {"B", "kB", "MB", "GB", "TB", "PB", "EB"};

    double float_size = size;

    int count = 0;

    while (float_size > max && count + 1 < (int)prefixes.size())
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
