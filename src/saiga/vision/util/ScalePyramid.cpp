/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */



#include "ScalePyramid.h"

#include "saiga/core/util/assert.h"
#include "saiga/core/util/tostring.h"

namespace Saiga
{
ScalePyramid::ScalePyramid(int levels, ScalePyramid::T scale_factor) : levels(levels), scale_factor(scale_factor)
{
    SAIGA_ASSERT(levels > 0);
    SAIGA_ASSERT(scale_factor > 0);

    log_scale_factor = log(scale_factor);

    scale_per_level.resize(levels);
    inv_scale_per_level.resize(levels);
    squared_scale_per_level.resize(levels);
    inv_squared_scale_per_level.resize(levels);

    scale_per_level[0]             = 1;
    inv_scale_per_level[0]         = 1;
    squared_scale_per_level[0]     = 1;
    inv_squared_scale_per_level[0] = 1;

    for (int i = 1; i < levels; ++i)
    {
        scale_per_level[i]     = scale_factor * scale_per_level[i - 1];
        inv_scale_per_level[i] = 1 / scale_per_level[i];

        squared_scale_per_level[i]     = scale_per_level[i] * scale_per_level[i];
        inv_squared_scale_per_level[i] = 1.f / squared_scale_per_level[i];
    }
}

std::ostream& operator<<(std::ostream& strm, const ScalePyramid& sp)
{
    strm << "[ScalePyramid] " << std::endl;
    strm << "Levels       : " << sp.levels << std::endl;
    strm << "Scale Factor : " << sp.scale_factor << std::endl;
    strm << "Scales       : " << to_string(sp.scale_per_level.begin(), sp.scale_per_level.end());
    return strm;
}


}  // namespace Saiga
