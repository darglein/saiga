/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once


#include "saiga/core/math/math.h"
#include "saiga/core/util/tostring.h"

#include "SimpleIni.h"

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
    return ini.GetAddString(section.c_str(), variable_name.c_str(), variable.c_str(), comment.c_str());
}

// template <>
inline double ReadWriteIni(Saiga::SimpleIni& ini, double variable, std::string section, std::string variable_name,
                           std::string comment)
{
    return ini.GetAddDouble(section.c_str(), variable_name.c_str(), variable, comment.c_str());
}

// template <>
inline long ReadWriteIni(Saiga::SimpleIni& ini, long variable, std::string section, std::string variable_name,
                         std::string comment)
{
    return ini.GetAddLong(section.c_str(), variable_name.c_str(), variable, comment.c_str());
}

inline long ReadWriteIni(Saiga::SimpleIni& ini, int variable, std::string section, std::string variable_name,
                         std::string comment)
{
    return ini.GetAddLong(section.c_str(), variable_name.c_str(), variable, comment.c_str());
}

// template <>
inline bool ReadWriteIni(Saiga::SimpleIni& ini, bool variable, std::string section, std::string variable_name,
                         std::string comment)
{
    return ini.GetAddBool(section.c_str(), variable_name.c_str(), variable, comment.c_str());
}

template <typename T>
inline std::vector<T> ReadWriteIniList(Saiga::SimpleIni& ini, std::vector<T> variable, std::string section,
                                       std::string variable_name, std::string comment = "", char sep = ',');

template <>
inline std::vector<std::string> ReadWriteIniList(Saiga::SimpleIni& ini, std::vector<std::string> variable,
                                                 std::string section, std::string variable_name, std::string comment,
                                                 char sep)
{
    return Saiga::split(
        ini.GetAddString(section.c_str(), variable_name.c_str(), Saiga::concat(variable, sep).c_str(), comment.c_str()),
        sep);
}


template <typename _Scalar, int _Rows, int _Cols>
std::string toIniString(const Eigen::Matrix<_Scalar, _Rows, _Cols>& M, char sep)
{
    std::string str;
    // Add entries to string, separated with ' ' in row major order.
    for (int i = 0; i < M.rows(); ++i)
        for (int j = 0; j < M.cols(); ++j) str += Saiga::to_string(M(i, j)) + sep;
    if (!str.empty())
    {
        // remove last sep
        str.pop_back();
    }

    return str;
}


template <typename _Scalar, int _Rows, int _Cols>
void fromIniString(const std::string& str, Eigen::Matrix<_Scalar, _Rows, _Cols>& M, char sep)
{
    auto arr = Saiga::split(str, sep);
    SAIGA_ASSERT((int)arr.size() == M.rows() * M.cols());

    for (int i = 0; i < M.rows(); ++i)
        for (int j = 0; j < M.cols(); ++j)
            M(i, j) = Saiga::FromStringConverter<_Scalar>::convert(arr[i * M.cols() + j]);
}

template <typename T, int rows, int cols>
inline Eigen::Matrix<T, rows, cols> ReadWriteIniList(Saiga::SimpleIni& ini, Eigen::Matrix<T, rows, cols> variable,
                                                     std::string section, std::string variable_name,
                                                     std::string comment, char sep)
{
    std::string str = toIniString(variable, sep);
    str             = ini.GetAddString(section.c_str(), variable_name.c_str(), str.c_str(), comment.c_str());
    fromIniString(str, variable, sep);
    return variable;
}

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
    _variable = ReadWriteIni(_ini, _variable, _section, #_variable, _comment)

#define INI_GETADD_LIST_COMMENT(_ini, _section, _variable, _sep, _comment) \
    _variable = ReadWriteIniList(_ini, _variable, _section, #_variable, _comment, _sep)

#define INI_GETADD(_ini, _section, _variable) INI_GETADD_COMMENT(_ini, _section, _variable, "")
