/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <sstream>
#include <string>
#include <vector>
#include <iomanip>
#include "saiga/config.h"
#include "saiga/util/ArrayView.h"

/**
 * This acts as a wrapper for std::to_string for build in data types.
 * For more complex data types the ostream operator << is used to create a string.
 * This allows for a simple conversion of vec3 -> string
 */

namespace Saiga
{
inline std::string to_string(int v) { return std::to_string(v); }
inline std::string to_string(long v) { return std::to_string(v); }
inline std::string to_string(long long v) { return std::to_string(v); }
inline std::string to_string(unsigned v) { return std::to_string(v); }
inline std::string to_string(unsigned long v) { return std::to_string(v); }
inline std::string to_string(unsigned long long v) { return std::to_string(v); }
inline std::string to_string(float v) { return std::to_string(v); }
inline std::string to_string(double v) { return std::to_string(v); }
inline std::string to_string(long double v) { return std::to_string(v); }
inline std::string to_string(const std::string& v) { return v; }
inline std::string to_string(const char* v) { return std::string(v); }

template <typename T>
inline std::string to_string(const T& v)
{
    std::stringstream sstream;
    sstream << v;
    return sstream.str();
}

/**
 * Saves the values in scientific notation with high precision.
 * Usefull for saving the value to a ascii file without losing precision.
 */
inline std::string to_string(double v, int precision)
{
    std::stringstream sstream;
    sstream << std::scientific << std::setprecision(precision) << v;
    return sstream.str();
}



inline float to_float(const std::string& str) { return std::atof(str.c_str()); }
inline double to_double(const std::string& str) { return std::atof(str.c_str()); }
inline int to_int(const std::string& str) { return std::atoi(str.c_str()); }
inline long int to_long(const std::string& str) { return std::atol(str.c_str()); }


template <typename T>
struct FromStringConverter
{
};
template <>
struct FromStringConverter<float>
{
    static float convert(const std::string& str) { return to_float(str); }
};
template <>
struct FromStringConverter<double>
{
    static float convert(const std::string& str) { return to_double(str); }
};
template <>
struct FromStringConverter<int>
{
    static float convert(const std::string& str) { return to_int(str); }
};
template <>
struct FromStringConverter<long>
{
    static float convert(const std::string& str) { return to_long(str); }
};



SAIGA_GLOBAL std::vector<std::string> split(const std::string& s, char delim);
SAIGA_GLOBAL std::string concat(const std::vector<std::string>& s, char delim);

SAIGA_GLOBAL std::string leadingZeroString(int number, int characterCount);
SAIGA_GLOBAL bool hasEnding(std::string const& fullString, std::string const& ending);

/**
 * @brief fileEnding
 * Extracts and returns the file ending of a string.
 * Example:
 *
 * "bla.jpg" -> "jpg"
 * "/usr/local/test.asdf" -> "asdf"
 */
SAIGA_GLOBAL std::string fileEnding(std::string const& str);
SAIGA_GLOBAL std::string removeFileEnding(std::string const& str);


template <typename ArrayType>
SAIGA_TEMPLATE std::string array_to_string(const ArrayType& array, char sep = ' ')
{
    std::string res;
    for (unsigned int i = 0; i < array.size(); ++i)
    {
        res += to_string(array[i]);
        if (i < array.size() - 1) res += sep;
    }
    return res;
}


template <typename T>
SAIGA_TEMPLATE std::vector<T> string_to_array(const std::string& string, char sep = ' ')
{
    FromStringConverter<T> converter;

    auto strArray = split(string, sep);
    std::vector<T> res;
    for (auto& s : strArray)
    {
        res.push_back(converter.convert(s));
    }
    return res;
}

}  // namespace Saiga
