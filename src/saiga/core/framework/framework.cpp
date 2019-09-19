/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "framework.h"

#include "saiga/core/image/image.h"
#include "saiga/core/math/floatingPoint.h"
#include "saiga/core/model/ModelLoader.h"
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

namespace Saiga
{
bool initialized = false;
std::string shaderDir;


bool isShaderDirectory(const std::string& dir)
{
    Directory dirbase(dir);
    Directory dirgeo(dir + "/geometry");
    return dirbase.existsFile("colored_points.glsl") && dirgeo.existsFile("deferred_mvp_texture.glsl");
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
    Saiga::SimpleIni ini;
    ini.LoadFile(file.c_str());


    std::string comment =
        "# Multiple search pathes must be separated by '!'.\n"
        "# Example:\n"
        "# shaderDirectory = shader!/usr/local/share/saiga/shader!somepath/asdf/shader";

    char sep = '!';
    shaderDirectory =
        split(ini.GetAddString("saiga", "shaderDirectory", concat(shaderDirectory, sep).c_str(), comment.c_str()), sep);
    textureDirectory = split(ini.GetAddString("saiga", "textureDirectory", concat(textureDirectory, sep).c_str()), sep);
    modelDirectory   = split(ini.GetAddString("saiga", "modelDirectory", concat(modelDirectory, sep).c_str()), sep);
    fontDirectory    = split(ini.GetAddString("saiga", "fontDirectory", concat(fontDirectory, sep).c_str()), sep);
    dataDirectory    = split(ini.GetAddString("saiga", "dataDirectory", concat(dataDirectory, sep).c_str()), sep);
    mainThreadName   = ini.GetAddString("saiga", "mainThreadName", mainThreadName.c_str());
    logging_enabled  = ini.GetAddBool("saiga", "logging", logging_enabled);
    verbose_logging  = ini.GetAddLong("saiga", "verbose_logging", verbose_logging);
    if (ini.changed()) ini.SaveFile(file.c_str());
}



void printSaigaInfo()
{
    std::cout << ConsoleColor::BLUE;
    Table table({2, 18, 10, 1});

    std::cout << "============ SAIGA ============" << std::endl;
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
          << "Debug"
          <<
#ifndef NDEBUG
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

    std::cout << "===============================" << std::endl;
    std::cout.unsetf(std::ios_base::floatfield);
    std::cout << ConsoleColor::RESET;
}


void initSaigaSample()
{
    SaigaParameters saigaParameters;
    saigaParameters.shaderDirectory = {SAIGA_PROJECT_SOURCE_DIR "/shader"};

    std::string dataDir              = SAIGA_PROJECT_SOURCE_DIR "/data";
    saigaParameters.textureDirectory = {dataDir, dataDir + "/textures"};
    saigaParameters.modelDirectory   = {dataDir, dataDir + "/models"};
    saigaParameters.fontDirectory    = {dataDir, dataDir + "/fonts"};
    saigaParameters.dataDirectory    = {dataDir};

    saigaParameters.fromConfigFile("config.ini");

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
