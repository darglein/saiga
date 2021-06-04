/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/icp/ICPDepthMap.h"
#include "saiga/vision/util/Depthmap.h"

#include <vector>

namespace Saiga
{
namespace ICP
{
SAIGA_VISION_API void multiViewICPAlign(size_t N, const std::vector<std::pair<size_t, size_t>>& pairs,
                                        const std::vector<AlignedVector<Correspondence>>& corrs,
                                        AlignedVector<SE3>& guesses, int iterations);


}  // namespace ICP
}  // namespace Saiga
