/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/util/DataStructures/ArrayView.h"

#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#ifdef SAIGA_HAS_STRING_VIEW
#    include <string_view>
#endif

#if 0 && __has_include(<charconv> )
#    include <charconv>
#    define SAIGA_USE_SV_CONV
#endif

/**
 * This acts as a wrapper for std::to_string for build in data types.
 * For more complex data types the ostream operator << is used to create a string.
 * This allows for a simple conversion of vec3 -> string
 */

namespace Saiga
{
inline std::string to_string(int v)
{
    return std::to_string(v);
}
inline std::string to_string(long v)
{
    return std::to_string(v);
}
inline std::string to_string(long long v)
{
    return std::to_string(v);
}
inline std::string to_string(unsigned v)
{
    return std::to_string(v);
}
inline std::string to_string(unsigned long v)
{
    return std::to_string(v);
}
inline std::string to_string(unsigned long long v)
{
    return std::to_string(v);
}
inline std::string to_string(float v)
{
    return std::to_string(v);
}
inline std::string to_string(double v)
{
    return std::to_string(v);
}
inline std::string to_string(long double v)
{
    return std::to_string(v);
}
inline std::string to_string(const std::string& v)
{
    return v;
}
inline std::string to_string(const char* v)
{
    return std::string(v);
}

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



inline float to_float(const std::string& str)
{
    return std::atof(str.c_str());
}

inline int to_int(const std::string& str)
{
    return std::atoi(str.c_str());
}

#ifdef SAIGA_HAS_STRING_VIEW
inline double to_double(const std::string_view& str)
{
#    if 0
    double d;
    std::from_chars(str.data(),str.data()+str.size(),d);
    return d;
#    else
    return std::atof(std::string(str).c_str());
#    endif
}

inline double to_long(const std::string_view& str)
{
    return std::atol(std::string(str).c_str());
}
#else
inline double to_double(const std::string& str)
{
    return std::atof(str.c_str());
}
inline long int to_long(const std::string& str)
{
    return std::atol(str.c_str());
}
#endif

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



SAIGA_CORE_API std::vector<std::string> split(const std::string& s, char delim);
SAIGA_CORE_API std::string concat(const std::vector<std::string>& s, char delim);

SAIGA_CORE_API std::string leadingZeroString(int number, int characterCount);
SAIGA_CORE_API bool hasEnding(std::string const& fullString, std::string const& ending);
SAIGA_CORE_API bool hasPrefix(std::string const& fullString, std::string const& prefix);

/**
 * @brief fileEnding
 * Extracts and returns the file ending of a string.
 * Example:
 *
 * "bla.jpg" -> "jpg"
 * "/usr/local/test.asdf" -> "asdf"
 */
SAIGA_CORE_API std::string fileEnding(std::string const& str);
SAIGA_CORE_API std::string removeFileEnding(std::string const& str);


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

template <typename Iterator>
SAIGA_TEMPLATE std::string to_string(Iterator begin, Iterator end)
{
    std::string result = "{";
    while (begin != end)
    {
        result += to_string(*begin);
        ++begin;
        if (begin != end) result += ", ";
    }
    result += "}";
    return result;
}

template <typename MatrixType>
std::string MatrixToString(const MatrixType& M)
{
    std::string result;
    char sep = ',';
    for (int i = 0; i < M.rows(); ++i)
    {
        for (int j = 0; j < M.cols(); ++j)
        {
            result += std::to_string(M(i, j)) + sep;
        }
    }
    // remove last sep
    if (M.rows() * M.cols() > 0) result.pop_back();
    return result;
}


template <typename MatrixType>
void StringToMatrix(std::string input, MatrixType& M)
{
    char sep   = ',';
    auto array = split(input, sep);
    SAIGA_ASSERT(array.size() == M.rows() * M.cols());
    for (int i = 0; i < M.rows(); ++i)
    {
        for (int j = 0; j < M.cols(); ++j)
        {
            M(i, j) = to_double(array[i * M.cols() + j]);
        }
    }
}

/**
 * Convert a byte size to a string with SI-prefix. e.g.: 512 -> "512 B" and 1536 -> "1.5 kB"
 * @param size Size to convert
 * @param base Base to calculate. Useful for HDD and SDD where the base is usually 1000.
 * @param max The displayed value will be in the range [0;max].
 * @param sep Separator between the value and the unit.
 * @param precision Precision for the value to display, if the value is in bytes, a precision of 0 will be used instead.
 * @return Nicely formatted SI-prefixed string.
 */
SAIGA_CORE_API std::string sizeToString(size_t size, size_t base = 1024, size_t max = 1536, const char* sep = " ",
                                        std::streamsize precision = 1);


#ifdef SAIGA_HAS_STRING_VIEW
struct SAIGA_TEMPLATE StringViewParser
{
    StringViewParser(std::string_view delims = " ,\n", bool allowDoubleDelims = true)
        : delims(delims), allowDoubleDelims(allowDoubleDelims)
    {
    }
    std::string_view next()
    {
        auto it = currentView.begin();
        while (it != currentView.end() && !isDelim(*it))
        {
            ++it;
        }
        auto result = currentView.substr(0, it - currentView.begin());
        currentView = currentView.substr(it - currentView.begin());
        advance();
        return result;
    }
    void set(std::string_view v)
    {
        currentView = v;
        advance();
    }

   private:
    std::string_view currentView;
    std::string_view delims;
    bool allowDoubleDelims = true;
    // Skip over all delims
    inline void advance()
    {
        auto it = currentView.begin();
        while (it != currentView.end() && isDelim(*it))
        {
            ++it;

            if (!allowDoubleDelims) break;
        }
        currentView = currentView.substr(it - currentView.begin());
    }
    inline bool isDelim(char c)
    {
        for (auto d : delims)
            if (d == c) return true;
        return false;
    }
};
#endif

}  // namespace Saiga
