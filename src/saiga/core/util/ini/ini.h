/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once


#include "saiga/core/math/Quaternion.h"
#include "saiga/core/math/Types.h"
#include "saiga/core/util/string_to_path.h"
#include "saiga/core/util/tostring.h"

#include "SimpleIni.h"

#include <filesystem>
#include <limits>
#include <stdexcept>

#include <type_traits>

namespace Saiga
{
namespace IniInternal
{
// Splits a list value. Surrounding whitespace is ignored, and with ' ' as separator so are repeated spaces.
inline std::vector<std::string> SplitList(const std::string& str, char sep)
{
    std::vector<std::string> result;
    for (auto& s : Saiga::split(str, sep))
    {
        auto begin = s.find_first_not_of(" \t\r\n");
        if (begin == std::string::npos)
        {
            if (sep == ' ') continue;
            s.clear();
        }
        else
        {
            s = s.substr(begin, s.find_last_not_of(" \t\r\n") - begin + 1);
        }
        result.push_back(s);
    }
    return result;
}

template <typename T>
std::string ValueToString(T v)
{
    if constexpr (std::is_floating_point<T>::value)
    {
        return NumberToStringRoundtrip(v);
    }
    else
    {
        return std::to_string(v);
    }
}

template <typename T>
T ParseValue(const std::string& str, const std::string& section, const std::string& name)
{
    using ParseType = std::conditional_t<std::is_floating_point<T>::value, T, long long>;
    ParseType value;
    if (!ParseNumber(str, value) || (std::is_integral<T>::value && (value < ParseType(std::numeric_limits<T>::min()) ||
                                                                    value > ParseType(std::numeric_limits<T>::max()))))
    {
        throw std::runtime_error("Invalid value '" + str + "' in [" + section + "] " + name);
    }
    return T(value);
}
}  // namespace IniInternal

/**
 *  ============== Usage Example ===============
 * Note:    The "getAdd" functions add the keys if they don't exist yet.
 *          This is usefull to auto-generate a config file if it so present.
 *
 *   // Checkout the full source at samples/ini/main.cpp
 *   std::string name;
 *   int w,h;
 *   bool b;
 *
 *   Saiga::SimpleIni ini;
 *   ini.LoadFile("config.ini");
 *
 *   name = ini.GetAddString ("window","name","Test Window");
 *   w    = ini.GetAddLong   ("window","width",1280);
 *   h    = ini.GetAddDouble ("window","height",720);
 *   b    = ini.GetAddBool   ("window","Fullscreen",false);
 *
 *   if(ini.changed()) ini.SaveFile("config.ini");
 *
 *   std::cout << name << " " << w << "x" << h << " " << b << std::endl;
 *
 */
//
// template <typename T>
// inline T ReadWriteIni(Saiga::SimpleIni& ini, T variable, std::string section, std::string variable_name,
//                      std::string comment = "");
//
// template <>
inline std::string ReadWriteIni(Saiga::SimpleIni& ini, std::string variable, std::string section,
                                std::string variable_name, std::string comment)
{
    return ini.GetAddString(section.c_str(), variable_name.c_str(), variable.c_str(),
                            comment.length() == 0 ? nullptr : comment.c_str());
}

inline std::filesystem::path ReadWriteIni(Saiga::SimpleIni& ini, std::filesystem::path variable, std::string section,
    std::string variable_name, std::string comment)
{
    std::string str = ReadWriteIni(ini, variable.u8string(), section, variable_name, comment);
    return str.empty() ? "" : string_to_path(str);
}

// template <>
inline double ReadWriteIni(Saiga::SimpleIni& ini, double variable, std::string section, std::string variable_name,
                           std::string comment)
{
    return ini.GetAddDouble(section.c_str(), variable_name.c_str(), variable,
                            comment.length() == 0 ? nullptr : comment.c_str());
}

// template <>
inline int64_t ReadWriteIni(Saiga::SimpleIni& ini, int64_t variable, std::string section, std::string variable_name,
                            std::string comment)
{
    return ini.GetAddLong(section.c_str(), variable_name.c_str(), variable,
                          comment.length() == 0 ? nullptr : comment.c_str());
}

inline int64_t ReadWriteIni(Saiga::SimpleIni& ini, int variable, std::string section, std::string variable_name,
                            std::string comment)
{
    return ini.GetAddLong(section.c_str(), variable_name.c_str(), variable,
                          comment.length() == 0 ? nullptr : comment.c_str());
}

// template <>
inline bool ReadWriteIni(Saiga::SimpleIni& ini, bool variable, std::string section, std::string variable_name,
                         std::string comment)
{
    return ini.GetAddBool(section.c_str(), variable_name.c_str(), variable,
                          comment.length() == 0 ? nullptr : comment.c_str());
}

template <typename T>
inline std::vector<T> ReadWriteIniList(Saiga::SimpleIni& ini, std::vector<T> variable, std::string section,
                                       std::string variable_name, std::string comment = "", char sep = ',');

template <>
inline std::vector<std::string> ReadWriteIniList(Saiga::SimpleIni& ini, std::vector<std::string> variable,
                                                 std::string section, std::string variable_name, std::string comment,
                                                 char sep)
{
    return Saiga::split(ini.GetAddString(section.c_str(), variable_name.c_str(), Saiga::concat(variable, sep).c_str(),
                                         comment.length() == 0 ? nullptr : comment.c_str()),
                        sep);
}

template <>
inline std::vector<std::filesystem::path> ReadWriteIniList(Saiga::SimpleIni& ini, std::vector<std::filesystem::path> variable,
    std::string section, std::string variable_name, std::string comment,
    char sep)
{
    std::vector<std::string> tmp;
    for (const auto& v : variable)
    {
        tmp.push_back(v.u8string());
    }

    tmp = Saiga::split(ini.GetAddString(section.c_str(), variable_name.c_str(), Saiga::concat(tmp, sep).c_str(),
        comment.length() == 0 ? nullptr : comment.c_str()),
        sep);

    std::vector<std::filesystem::path> result;
    for (const auto& v : tmp)
    {
        result.push_back(string_to_path(v));
    }
    return result;
}

template <>
inline std::vector<double> ReadWriteIniList(Saiga::SimpleIni& ini, std::vector<double> variable, std::string section,
                                            std::string variable_name, std::string comment, char sep)
{
    std::vector<std::string> tmp;
    for (const auto& v : variable)
    {
        tmp.push_back(IniInternal::ValueToString(v));
    }

    tmp =
        IniInternal::SplitList(ini.GetAddString(section.c_str(), variable_name.c_str(), Saiga::concat(tmp, sep).c_str(),
                                                comment.length() == 0 ? nullptr : comment.c_str()),
                               sep);

    std::vector<double> result;
    for (const auto& v : tmp)
    {
        result.push_back(IniInternal::ParseValue<double>(v, section, variable_name));
    }
    return result;
}


template <>
inline std::vector<int> ReadWriteIniList(Saiga::SimpleIni& ini, std::vector<int> variable, std::string section,
                                         std::string variable_name, std::string comment, char sep)
{
    std::vector<std::string> tmp;
    for (const auto& v : variable)
    {
        tmp.push_back(std::to_string(v));
    }

    tmp =
        IniInternal::SplitList(ini.GetAddString(section.c_str(), variable_name.c_str(), Saiga::concat(tmp, sep).c_str(),
                                                comment.length() == 0 ? nullptr : comment.c_str()),
                               sep);

    std::vector<int> result;
    for (const auto& v : tmp)
    {
        result.push_back(IniInternal::ParseValue<int>(v, section, variable_name));
    }
    return result;
}

template <typename _Scalar, int _Rows, int _Cols>
std::string toIniString(const Matrix<_Scalar, _Rows, _Cols>& M, char sep)
{
    std::string str;
    // Add entries to string, separated with ' ' in row major order.
    for (int i = 0; i < M.rows(); ++i)
        for (int j = 0; j < M.cols(); ++j) str += IniInternal::ValueToString(M(i, j)) + sep;
    if (!str.empty())
    {
        // remove last sep
        str.pop_back();
    }

    return str;
}


// Throws std::runtime_error if the number of values does not match or a value is not a number.
template <typename _Scalar, int _Rows, int _Cols>
void fromIniString(const std::string& str, Matrix<_Scalar, _Rows, _Cols>& M, char sep, const std::string& section = "",
                   const std::string& name = "")
{
    auto arr = IniInternal::SplitList(str, sep);
    if ((int)arr.size() != M.rows() * M.cols())
    {
        throw std::runtime_error("Expected " + std::to_string(M.rows() * M.cols()) + " values in [" + section + "] " +
                                 name + " but got " + std::to_string(arr.size()) + ": '" + str + "'");
    }

    for (int i = 0; i < M.rows(); ++i)
        for (int j = 0; j < M.cols(); ++j)
            M(i, j) = IniInternal::ParseValue<_Scalar>(arr[i * M.cols() + j], section, name);
}

template <typename T, int rows, int cols>
inline Matrix<T, rows, cols> ReadWriteIniList(Saiga::SimpleIni& ini, Matrix<T, rows, cols> variable,
                                                     std::string section, std::string variable_name,
                                                     std::string comment, char sep)
{
    std::string str = toIniString(variable, sep);
    str             = ini.GetAddString(section.c_str(), variable_name.c_str(), str.c_str(),
                           comment.length() == 0 ? nullptr : comment.c_str());
    fromIniString(str, variable, sep, section, variable_name);
    return variable;
}

template <typename T>
inline Quaternion<T> ReadWriteIniList(Saiga::SimpleIni& ini, Quaternion<T> variable,
    std::string section, std::string variable_name,
    std::string comment, char sep)
{
    Matrix<T, 4, 1> coeffs = variable.coeffs();
    coeffs = ReadWriteIniList(ini, coeffs, section, variable_name, comment, sep);
    variable = Quaternion<T>(coeffs);
    return variable;
}

}  // namespace Saiga

/**
 * Helper macros for creating the most common use-case:
 *
 * - The variable x is loaded from the ini file
 * - The default value is the previous value of x
 * - The name in the .ini file is the actual variable name
 *
 * Example usage:
 *
 * double foo = 3.14;
 * INI_GETADD_DOUBLE(ini, "Math", foo);
 */
#define INI_GETADD_COMMENT(_ini, _section, _variable, _comment) \
    _variable = Saiga::ReadWriteIni(_ini, _variable, _section, #_variable, _comment)

#define INI_GETADD_LIST_COMMENT(_ini, _section, _variable, _sep, _comment) \
    _variable = Saiga::ReadWriteIniList(_ini, _variable, _section, #_variable, _comment, _sep)

#define INI_GETADD(_ini, _section, _variable) INI_GETADD_COMMENT(_ini, _section, _variable, "")
