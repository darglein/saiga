/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */



#include "ScalePyramid.h"

#include "saiga/core/util/assert.h"
#include "saiga/core/util/tostring.h"

namespace Saiga
{
ScalePyramid::ScalePyramid(int _levels, ScalePyramid::T scale_factor, int total_features)
    : num_levels(_levels), total_num_features(total_features), scale_factor(scale_factor), levels(_levels)
{
    SAIGA_ASSERT(num_levels > 0);
    SAIGA_ASSERT(scale_factor > 0);

    log_scale_factor = log(scale_factor);
    levels.resize(num_levels);

    levels[0].scale             = 1;
    levels[0].inv_scale         = 1;
    levels[0].squared_scale     = 1;
    levels[0].inv_squared_scale = 1;


    T factor                   = 1.0 / scale_factor;
    T nDesiredFeaturesPerScale = total_num_features * (1 - factor) / (1 - std::pow(factor, num_levels));

    int sumFeatures = 0;

    for (int i = 0; i < num_levels - 1; ++i)
    {
        auto& level        = levels[i];
        level.num_features = Saiga::iRound(nDesiredFeaturesPerScale);
        sumFeatures += level.num_features;
        nDesiredFeaturesPerScale *= factor;
    }

    for (int i = 1; i < num_levels; ++i)
    {
        auto& level = levels[i];

        level.scale     = scale_factor * levels[i - 1].scale;
        level.inv_scale = 1 / level.scale;

        level.squared_scale     = level.scale * level.scale;
        level.inv_squared_scale = 1.f / level.squared_scale;
    }
    levels.back().num_features = std::max(total_num_features - sumFeatures, 0);
}

std::ostream& operator<<(std::ostream& strm, const ScalePyramid& sp)
{
    strm << "[ScalePyramid] " << std::endl;
    strm << "Levels       : " << sp.num_levels << std::endl;
    strm << "Scale Factor : " << sp.scale_factor << std::endl;
    strm << "Total Features : " << sp.total_num_features << std::endl;
    return strm;
}


}  // namespace Saiga
