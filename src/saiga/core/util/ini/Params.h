/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once


#include "saiga/core/math/Quaternion.h"
#include "saiga/core/math/Types.h"
#include "saiga/core/util/assert.h"
#include "saiga/core/util/commandLineArguments.h"
#include "saiga/core/util/ini/ParamsReduced.h"

#include "ini.h"


struct IniFileParamIterator
{
    Saiga::SimpleIni* ini;

    template <typename T>
    void SaigaParam(std::string section, T& variable, T default_value, std::string name, std::string comment = "")
    {
        variable = ReadWriteIni(*ini, variable, section, name, comment);
    }

    template <typename T>
    void SaigaParamList(std::string section, T& variable, T default_value, std::string name, char sep,
                        std::string comment = "")
    {
        // INI_GETADD_LIST_COMMENT(*ini, name.c_str(), variable, sep, comment);
        variable = ReadWriteIniList(*ini, variable, section, name, comment, sep);
    }
};

struct ApplicationParamIterator
{
    CLI::App* app;

    template <typename T>
    void SaigaParam(std::string section, T& variable, T default_value, std::string name, std::string comment = "")
    {
        app->add_option("--" + section + "." + name, variable, comment, true);
    }

    void SaigaParam(std::string section,
        std::filesystem::path& variable,
        std::filesystem::path default_value,
        std::string name,
        std::string comment)
    {
        auto call_back = [&variable](const std::string& result) 
        {
            variable = string_to_path(result);
            return true;
        };

        app->add_option_function<std::string>("--" + section + "." + name, call_back, comment);
    }

    template <typename T>
    void SaigaParamList(std::string section, T& variable, T default_value, std::string name, char sep,
                        std::string comment = "")
    {
        if (sep == ' ')
        {
            // app->add_option("--" + name, *variable, comment, true);
        }
    }

    template <typename _Scalar, int _Rows, int _Cols>
    void SaigaParamList(std::string section, Eigen::Matrix<_Scalar, _Rows, _Cols>& variable,
                        Eigen::Matrix<_Scalar, _Rows, _Cols> default_value, std::string name, char sep,
                        std::string comment = "")
    {
        if (sep == ' ')
        {
            auto call_back = [&variable](const std::vector<std::string>& result) -> bool
            {
                SAIGA_ASSERT(result.size() == _Rows * _Cols);
                for (int i = 0; i < _Rows; ++i)
                {
                    for (int j = 0; j < _Cols; ++j)
                    {
                        variable(i, j) = Saiga::to_double(result[i * _Cols + j]);
                    }
                }
                return true;
            };
            auto call_back2 = []() -> std::string { return ""; };

            CLI::Option* options = app->add_option("--" + section + "." + name, call_back, comment, true, call_back2);
            options->type_size(1);
            options->expected(_Rows * _Cols);
        }
    }

    template <typename _Scalar>
    void SaigaParamList(std::string section, Eigen::Quaternion<_Scalar>& variable,
                        Eigen::Quaternion<_Scalar> default_value, std::string name, char sep, std::string comment = "")
    {
        Eigen::Matrix<_Scalar, 4, 1> coeffs         = variable.coeffs();
        Eigen::Matrix<_Scalar, 4, 1> default_coeffs = default_value.coeffs();

        SaigaParamList(section, coeffs, default_coeffs, name, sep, comment);

        variable = Eigen::Quaternion<_Scalar>(coeffs);
    }

    template <typename T>
    void SaigaParamList(std::string section, std::vector<T>& variable, std::vector<T> default_value, std::string name,
                        char sep, std::string comment = "")
    {
        // if (sep == ' ')
        {
            app->add_option("--" + section + "." + name, variable, comment, true);
        }
    }

    void SaigaParamList(std::string section, std::vector<std::filesystem::path>& variable, std::vector<std::filesystem::path> default_value, std::string name,
        char sep, std::string comment)
    {
        // if (sep == ' ')
        {
            auto call_back = [&variable](const std::vector<std::string>& result)
            {
                variable.clear();
                variable.reserve(result.size());
                for (const std::string& s : result)
                {
                    variable.push_back(string_to_path(s));
                }
                return true;
            };

            app->add_option_function<std::vector<std::string>>("--" + section + "." + name, call_back, comment);
        }
    }
};

struct TablePrintParamIterator
{
    std::ostream& strm;
    int column_width = 15;

    TablePrintParamIterator(std::ostream& strm, int cw) : strm(strm), column_width(cw) {}

    template <typename T>
    void SaigaParam(std::string section, T& variable, T default_value, std::string name, std::string comment = "")
    {
        strm << std::left << std::setw(column_width) << name << std::right << variable << "\n";
    }

    template <typename T>
    void SaigaParamList(std::string section, T& variable, T default_value, std::string name, char sep,
                        std::string comment = "")
    {
        strm << std::left << std::setw(column_width) << name << std::right << "(skipped)" << "\n";
    }
};



#define SAIGA_PARAM_STRUCT_FUNCTIONS                              \
    void Load(CLI::App& app)                                      \
    {                                                             \
        ApplicationParamIterator appit;                           \
        appit.app = &app;                                         \
        Params(&appit);                                           \
    }                                                             \
                                                                  \
                                                                  \
    virtual void Load(const std::filesystem::path& file)          \
    {                                                             \
        Saiga::SimpleIni ini_;                                    \
        ini_.LoadFile(file.c_str());                              \
        IniFileParamIterator iniit;                               \
        iniit.ini = &ini_;                                        \
        Params(&iniit);                                           \
        if (ini_.changed()) ini_.SaveFile(file.c_str());          \
    }                                                             \
                                                                  \
                                                                  \
    virtual void Save(const std::filesystem::path& file)          \
    {                                                             \
        Saiga::SimpleIni ini_;                                    \
        ini_.LoadFile(file.c_str());                              \
        IniFileParamIterator iniit;                               \
        iniit.ini = &ini_;                                        \
        Params(&iniit);                                           \
        ini_.SaveFile(file.c_str());                              \
    }                                                             \
    virtual void Print(std::ostream& strm, int column_width = 30) \
    {                                                             \
        TablePrintParamIterator tableit(strm, column_width);      \
        strm << "[" << name_ << "]\n";                            \
        Params(&tableit);                                         \
    }


#define SAIGA_PARAM_STRUCT_FUNCTIONS_NAMED(name)             \
    void name::Load(CLI::App& app)                           \
    {                                                        \
        ApplicationParamIterator appit;                      \
        appit.app = &app;                                    \
        Params(&appit);                                      \
    }                                                        \
                                                             \
                                                             \
    void name::Load(const std::filesystem::path& file)       \
    {                                                        \
        Saiga::SimpleIni ini_;                               \
        ini_.LoadFile(file.c_str());                         \
        IniFileParamIterator iniit;                          \
        iniit.ini = &ini_;                                   \
        Params(&iniit);                                      \
        if (ini_.changed()) ini_.SaveFile(file.c_str());     \
    }                                                        \
                                                             \
                                                             \
    void name::Save(const std::filesystem::path& file)       \
    {                                                        \
        Saiga::SimpleIni ini_;                               \
        ini_.LoadFile(file.c_str());                         \
        IniFileParamIterator iniit;                          \
        iniit.ini = &ini_;                                   \
        Params(&iniit);                                      \
        ini_.SaveFile(file.c_str());                         \
    }                                                        \
    void name::Print(std::ostream& strm, int column_width)   \
    {                                                        \
        TablePrintParamIterator tableit(strm, column_width); \
        strm << "[" << name_ << "]\n";                       \
        Params(&tableit);                                    \
    }
