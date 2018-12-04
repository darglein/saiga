/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/geometry/plane.h"

#include "internal/noGraphicsAPI.h"
namespace Saiga
{
std::ostream& operator<<(std::ostream& os, const Plane& pl)
{
    os << "x * " << pl.normal << " - " << pl.d << " = 0";
    return os;
}

}  // namespace Saiga
