/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "arguments.h"

#include "saiga/core/util/ini/ini.h"

namespace Saiga
{
void Arguments::Load(const std::string& file, bool write_missing_values)
{
    Saiga::SimpleIni ini;
    ini.LoadFile(file.c_str());

    if (ini.changed()) ini.SaveFile(file.c_str());
}
void Arguments::RegisterArgument(const std::string& name, float& f) {}
}  // namespace Saiga
