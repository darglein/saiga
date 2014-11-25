#pragma once
#include "libhello/util/glm.h"
#include "libhello/geometry/aabb.h"
#include "libhello/geometry/ray.h"
#include "libhello/geometry/triangle.h"

using glm::min;
using glm::max;


class Raytracer
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

    //writes all found intersections to output and returns number
    int trace(Ray &r, std::vector<Result> &output);
};

inline bool operator< (const Raytracer::Result& lhs, const Raytracer::Result& rhs){ return lhs.distance<rhs.distance; }
inline bool operator> (const Raytracer::Result& lhs, const Raytracer::Result& rhs){return rhs < lhs;}
inline bool operator<=(const Raytracer::Result& lhs, const Raytracer::Result& rhs){return !(lhs > rhs);}
inline bool operator>=(const Raytracer::Result& lhs, const Raytracer::Result& rhs){return !(lhs < rhs);}
