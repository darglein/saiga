/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <saiga/config.h>
#include "saiga/util/glm.h"
#include "saiga/geometry/aabb.h"
#include "saiga/geometry/sphere.h"
#include "saiga/geometry/triangle.h"
#include "saiga/geometry/plane.h"
#include "saiga/geometry/ray.h"

//TODO:
//move intersection methods from other classes to this file

namespace Saiga {
namespace Intersection {

/**
 * Intersection of 2 planes.
 * 2 general planes intersect in a line given by outDir and outPoint, unless they are parallel.
 * Returns false if the planes are parallel.
 */
SAIGA_GLOBAL bool PlanePlane(const Plane& p1, const Plane& p2, Ray& outRay);

/**
 * Intersection of a ray with a sphere.
 * There are either 2 intersections or 0, given by the return value.
 * t2 is always greater or equal to t1
 */
SAIGA_GLOBAL bool RaySphere(const Ray& ray, const Sphere &sphere, float &t1, float &t2);

}
}
