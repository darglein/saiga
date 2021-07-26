/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"
#include "saiga/core/geometry/Frustum.h"

#include "gtest/gtest.h"

namespace Saiga
{
TEST(Frustum, frustum)
{
    Frustum frustum(mat4::Identity(), radians(60.f), 1, 1, 2);
    std::cout << frustum << std::endl;
}


}  // namespace Saiga
