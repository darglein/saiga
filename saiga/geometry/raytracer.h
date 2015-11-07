#pragma once

#include <saiga/config.h>
#include "saiga/util/glm.h"
#include "saiga/geometry/aabb.h"
#include "saiga/geometry/ray.h"
#include "saiga/geometry/triangle.h"

#include <vector>

using glm::min;
using glm::max;


class SAIGA_GLOBAL Raytracer
{
public:
	std::vector<Triangle> &triangles;
    Raytracer(std::vector<Triangle> &triangles):triangles(triangles){}

    struct Result{
        bool valid;
        bool back;
        float distance;
        unsigned int triangle;

    };

    //find closest intersection
    Result trace(Ray &r);
	Raytracer& operator=(const Raytracer&) = delete;

    //writes all found intersections to output and returns number
    int trace(Ray &r, std::vector<Result> &output);
};

inline bool operator< (const Raytracer::Result& lhs, const Raytracer::Result& rhs){ return lhs.distance<rhs.distance; }
inline bool operator> (const Raytracer::Result& lhs, const Raytracer::Result& rhs){return rhs < lhs;}
inline bool operator<=(const Raytracer::Result& lhs, const Raytracer::Result& rhs){return !(lhs > rhs);}
inline bool operator>=(const Raytracer::Result& lhs, const Raytracer::Result& rhs){return !(lhs < rhs);}
