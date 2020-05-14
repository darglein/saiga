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
    INI_GETADD_DOUBLE(ini, group, playback_fps);
    INI_GETADD_STRING(ini, group, dir);
    //    INI_GETADD_STRING(ini, group, groundTruth);
    INI_GETADD_LONG(ini, group, startFrame);
    INI_GETADD_LONG(ini, group, maxFrames);
    INI_GETADD_BOOL(ini, group, multiThreadedLoad);
    INI_GETADD_BOOL(ini, group, only_first_image);
    INI_GETADD_BOOL(ini, group, preload);
    INI_GETADD_BOOL(ini, group, force_monocular);
    if (ini.changed()) ini.SaveFile(file.c_str());
}

std::ostream& operator<<(std::ostream& strm, const DatasetParameters& params)
{
    strm << params.dir << std::endl;
    return strm;
}
}  // namespace Saiga
