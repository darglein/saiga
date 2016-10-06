#pragma once

#include <saiga/config.h>
#include "saiga/util/glm.h"
#include "saiga/geometry/aabb.h"

class SAIGA_GLOBAL Sphere
{
public:
    vec3 pos;
    float r;


    Sphere(void){}

    Sphere(const vec3 &p, float r) :pos(p),r(r){}
    ~Sphere(void){}




    int intersectAabb(const aabb &other);
    bool intersectAabb2(const aabb &other);

    void getMinimumAabb(aabb &box);

    bool contains(vec3 p);
    bool intersect(const Sphere &other);

//    TriangleMesh* createMesh(int rings, int sectors);
//    void addToBuffer(std::vector<VertexNT> &vertices, std::vector<GLuint> &indices, int rings, int sectors);
};

