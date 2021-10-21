/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "framework.h"

#include "saiga/core/image/image.h"
#include "saiga/core/math/floatingPoint.h"
#include "saiga/core/util/ConsoleColor.h"
#include "saiga/core/util/Thread/threadName.h"
#include "saiga/core/util/assert.h"
#include "saiga/core/util/crash.h"
#include "saiga/core/util/directory.h"
#include "saiga/core/util/easylogging++.h"
#include "saiga/core/util/fileChecker.h"
#include "saiga/core/util/ini/ini.h"
#include "saiga/core/util/table.h"
#include "saiga/core/util/tostring.h"
#include "saiga/saiga_git_sha1.h"

#include "glog/logging.h"


namespace Saiga
{
bool initialized = false;
std::string shaderDir;


bool isShaderDirectory(const std::string& dir)
{
    Directory dirbase(dir);
    Directory dirgeo(dir + "/geometry");
    return dirbase.existsFile("imgui_gl.glsl") && dirgeo.existsFile("deferred_mvp_texture.glsl");
}

bool findShaders(const SaigaParameters& params)
{
    std::vector<std::string> searchPathes = {
        // First check in the local working directory
        "shader",
    };

    searchPathes.insert(searchPathes.end(), params.shaderDirectory.begin(), params.shaderDirectory.end());
    // Then the given paramter from the config file
    //        params.shaderDirectory,
    // And last the install prefix from cmake
    searchPathes.push_back(SAIGA_INSTALL_PREFIX "/share/saiga/shader");
    searchPathes.push_back(SAIGA_SHADER_PREFIX);


    for (auto str : searchPathes)
    {
        if (isShaderDirectory(str))
        {
            shaderDir = str;
            break;
        }
    }

    if (shaderDir.empty())
    {
        std::cout << "Could not find the Saiga shaders." << std::endl;
        std::cout << "Set the 'shaderDirectory' variable of 'SaigaParameters' accordingly." << std::endl;
        for (auto s : searchPathes)
        {
            std::cout << "     " << s << std::endl;
        }
        return false;
    }
    return true;
}


void SaigaParameters::fromConfigFile(const std::string& file)
{
    CHECK_NE(1, 2) << ": The world must be ending!";

    Saiga::SimpleIni ini;
    ini.LoadFile(file.c_str());


    std::string comment =
        "# Multiple search pathes must be separated by '!'.\n"
        "# Example:\n"
        "# shaderDirectory = shader!/usr/local/share/saiga/shader!somepath/asdf/shader";

    char sep = '!';
    INI_GETADD_LIST_COMMENT(ini, "saiga", shaderDirectory, sep, comment.c_str());
    INI_GETADD_LIST_COMMENT(ini, "saiga", modelDirectory, sep, comment.c_str());
    INI_GETADD_LIST_COMMENT(ini, "saiga", fontDirectory, sep, comment.c_str());
    INI_GETADD_LIST_COMMENT(ini, "saiga", textureDirectory, sep, comment.c_str());
    INI_GETADD_LIST_COMMENT(ini, "saiga", dataDirectory, sep, comment.c_str());

    INI_GETADD(ini, "saiga", mainThreadName);
    INI_GETADD(ini, "saiga", logging_enabled);
    INI_GETADD(ini, "saiga", verbose_logging);

    if (ini.changed()) ini.SaveFile(file.c_str());
}



void printSaigaInfo()
{
#ifdef CMAKE_RELWITHDEBINFO
    auto cmake_build_type = "RelWithDebInfo";
#endif
#ifdef CMAKE_DEBUG
    auto cmake_build_type = "Debug";
#endif
#ifdef CMAKE_RELEASE
    auto cmake_build_type = "Release";
#endif
#ifdef CMAKE_MINSIZEREL
    auto cmake_build_type = "MinSizeRel";
#endif


    std::cout << ConsoleColor::BLUE;
    Table table({2, 18, 16, 1});

    std::cout << "Ref. " << SAIGA_GIT_SHA1 << std::endl;
    std::cout << "=============== Saiga ===============" << std::endl;
    table << "|"
          << "Saiga Version" << SAIGA_VERSION << "|";
    table << "|"
          << "Eigen Version"
          << (to_string(EIGEN_WORLD_VERSION) + "." + to_string(EIGEN_MAJOR_VERSION) + "." +
              to_string(EIGEN_MINOR_VERSION))
          << "|";

    table << "|"
          << "Compiler" << (SAIGA_COMPILER_STRING) << "|";
    table << "|"
          << "  -> Version" << SAIGA_COMPILER_VERSION << "|";
    table << "|"
          << "Build Type" << cmake_build_type << "|";
    table << "|"
          << "Debug"
          <<
#ifndef NDEBUG
        "1"
#else
        "0"
#endif
          << "|";
    table << "|"
          << "Eigen Debug"
          <<
#ifndef EIGEN_NO_DEBUG
        "1"
#else
        "0"
#endif
          << "|";

    table << "|"
          << "ASAN"
          <<
#ifdef SAIGA_DEBUG_ASAN
        "1"
#else
        "0"
#endif
          << "|";

    table << "|"
          << "Asserts"
          <<
#ifdef SAIGA_ASSERTS
        "1"
#else
        "0"
#endif
          << "|";
    table << "|"
          << "Optimizations"
          <<
#ifdef SAIGA_FULL_OPTIMIZE
        "1"
#else
        "0"
#endif
          << "|";
    std::cout << "=====================================" << std::endl;

    std::cout.unsetf(std::ios_base::floatfield);
    std::cout << ConsoleColor::RESET;

    //    std::cout << "Pathes" << std::endl;
    //    std::cout << "SAIGA_PROJECT_SOURCE_DIR " << SAIGA_PROJECT_SOURCE_DIR << std::endl;
    //    std::cout << SearchPathes::shader << std::endl;
}


void initSaigaSample()
{
    SaigaParameters saigaParameters;
    saigaParameters.fromConfigFile("config.ini");
    saigaParameters.shaderDirectory.push_back(SAIGA_PROJECT_SOURCE_DIR "/shader");


    std::string dataDir = SAIGA_PROJECT_SOURCE_DIR "/data";
    saigaParameters.textureDirectory.push_back(dataDir);
    saigaParameters.modelDirectory.push_back(dataDir);
    saigaParameters.fontDirectory.push_back(dataDir);
    saigaParameters.dataDirectory.push_back(dataDir);

    saigaParameters.textureDirectory.push_back(dataDir + "/textures");
    saigaParameters.modelDirectory.push_back(dataDir + "/models");
    saigaParameters.fontDirectory.push_back(dataDir + "/fonts");

    initSaiga(saigaParameters);
    catchSegFaults();
}

void initSaiga(const SaigaParameters& params)
{
    if (initialized)
    {
        return;
    }


    FP::resetSSECSR();

    bool gotShaders = findShaders(params);

    if (!gotShaders) exit(1);


    SearchPathes::shader.addSearchPath(shaderDir);
    SearchPathes::shader.addSearchPath(shaderDir + "/include");
    SearchPathes::shader.addSearchPath(params.shaderDirectory);

    SearchPathes::image.addSearchPath(params.textureDirectory);
    SearchPathes::data.addSearchPath(params.dataDirectory);
    SearchPathes::font.addSearchPath(params.fontDirectory);
    SearchPathes::model.addSearchPath(params.modelDirectory);

    setThreadName(params.mainThreadName);

#if 0
    el::Configurations defaultConf;
    defaultConf.setToDefault();
    // Values are always std::string
    defaultConf.set(el::Level::Global, el::ConfigurationType::ToFile, "false");
    // defaultConf.set(el::Level::Global, el::ConfigurationType::ToStandardOutput,
    // std::to_string(params.logging_enabled));
    defaultConf.set(el::Level::Global, el::ConfigurationType::Enabled, std::to_string(params.logging_enabled));
    //    defaultConf.set(el::Level::Verbose, el::ConfigurationType::Enabled, std::to_string(params.verbose_logging));


    if (params.logging_enabled && params.verbose_logging > 0)
    {
        defaultConf.set(el::Level::Verbose, el::ConfigurationType::Enabled, "true");
        el::Loggers::setVerboseLevel(params.verbose_logging);
    }
    else
    {
        defaultConf.set(el::Level::Verbose, el::ConfigurationType::Enabled, "false");
    }

    el::Loggers::reconfigureLogger("default", defaultConf);
#endif

    printSaigaInfo();

    initialized = true;
}

void cleanupSaiga()
{
    initialized = false;
}

void initSaigaSampleNoWindow(bool createConfig)
{
    Saiga::initSaigaSample();
    Saiga::SaigaParameters saigaParameters;
    if (createConfig) saigaParameters.fromConfigFile("config.ini");
    initSaiga(saigaParameters);
    catchSegFaults();
}



}  // namespace Saiga
