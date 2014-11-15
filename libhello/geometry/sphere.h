#pragma once

#include "libhello/util/glm.h"
#include "libhello/geometry/aabb.h"

class Sphere
{
public:
    glm::vec3 pos;
    float r;


    Sphere(void){}

    Sphere(const vec3 &p, float r) :pos(p),r(r){}
    ~Sphere(void){}




    int intersectAabb(const aabb &other);
    bool intersectAabb2(const aabb &other);

    void getMinimumAabb(aabb &box);

    bool contains(vec3 p);

//    TriangleMesh* createMesh(int rings, int sectors);
//    void addToBuffer(std::vector<VertexNT> &vertices, std::vector<GLuint> &indices, int rings, int sectors);
};

