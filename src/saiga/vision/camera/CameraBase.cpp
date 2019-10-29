/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "CameraBase.h"

#include "saiga/core/Core.h"
namespace Saiga
{
void DatasetParameters::fromConfigFile(const std::string& file)
{
    Saiga::SimpleIni ini;
    ini.LoadFile(file.c_str());

    auto group = "Dataset";
    INI_GETADD_DOUBLE(ini, group, fps);
    INI_GETADD_STRING(ini, group, dir);
    INI_GETADD_STRING(ini, group, groundTruth);
    INI_GETADD_LONG(ini, group, startFrame);
    INI_GETADD_LONG(ini, group, maxFrames);
    INI_GETADD_BOOL(ini, group, multiThreadedLoad);
    if (ini.changed()) ini.SaveFile(file.c_str());
}
}  // namespace Saiga
