/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "ray.h"

#include "internal/noGraphicsAPI.h"

#include <iostream>
namespace Saiga
{
std::ostream& operator<<(std::ostream& os, const Ray& r)
{
    os << "[Ray] " << r.origin.transpose() << " | " << r.direction.transpose();
    return os;
}

}  // namespace Saiga
