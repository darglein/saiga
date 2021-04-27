/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "env.h"

#include "internal/noGraphicsAPI.h"

#include <cstdlib>

namespace Saiga
{
int SetEnv(const std::string& name, const std::string& value, int replace)
{
#if defined(_WIN32)
    int errcode = 0;
    if (!replace)
    {
        size_t envsize = 0;
        errcode        = getenv_s(&envsize, NULL, 0, name.c_str());
        if (errcode || envsize) return errcode;
    }
    return _putenv_s(name.c_str(), value.c_str());
#else

    return setenv(name.c_str(), value.c_str(), replace);
#endif
}

std::string GetEnv(const std::string& name)
{
    return std::string(getenv(name.c_str()));
}


}  // namespace Saiga
