/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "plane.h"

#include "internal/noGraphicsAPI.h"
namespace Saiga
{
std::ostream& operator<<(std::ostream& os, const Plane& pl)
{
    os << "n = (" << pl.normal.transpose() << ")   d = " << pl.d;
    return os;
}

}  // namespace Saiga
