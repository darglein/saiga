/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#ifndef _USE_MATH_DEFINES
#    define _USE_MATH_DEFINES
#endif

#include "triangle_mesh_generator.h"

#include "internal/noGraphicsAPI.h"
namespace Saiga
{
typedef TriangleMesh<VertexNT, uint32_t> default_mesh_t;

#if 0
std::shared_ptr<default_mesh_t> TriangleMeshGenerator::UVSphereMesh(const Sphere& sphere, int rings, int sectors)
{
    default_mesh_t* mesh = new default_mesh_t();
    float const R        = 1.f / (float)(rings);
    float const S        = 1.f / (float)(sectors);

    for (int r = 0; r < rings + 1; r++)
    {
        for (int s = 0; s < sectors; s++)
        {
            float y = sphere.r * sin(-M_PI_2 + M_PI * r * R);
            float x = sphere.r * cos(2 * M_PI * s * S) * sin(M_PI * r * R);
            float z = sphere.r * sin(2 * M_PI * s * S) * sin(M_PI * r * R);

            VertexNT vert;
            vert.texture  = vec2(s * S, r * R);
            vert.position = vec4(x, y, z, 1);
            vert.normal   = vec4(x, y, z, 0);
            mesh->vertices.push_back(vert);
        }
    }


    for (int r = 0; r < rings; r++)
    {
        for (int s = 0; s < sectors; s++)
        {
            if (r != rings - 1)
            {
                Face face;
                face.v1 = (r + 1) * sectors + s;
                face.v2 = (r + 1) * sectors + (s + 1) % sectors;
                face.v3 = r * sectors + (s + 1) % sectors;
                mesh->faces.push_back(face);
            }
            if (r != 0)
            {
                Face face;
                face.v1 = (r + 1) * sectors + s;
                face.v2 = r * sectors + (s + 1) % sectors;
                face.v3 = r * sectors + s;
                mesh->faces.push_back(face);
            }
        }
    }

    return std::shared_ptr<default_mesh_t>(mesh);
}

std::shared_ptr<default_mesh_t> TriangleMeshGenerator::IcoSphereMesh(const Sphere& sphere, int resolution)
{
    (void)sphere;
    default_mesh_t* mesh = new default_mesh_t();
    float t              = (1.0 + sqrt(5.0)) / 2.0;


    mesh->vertices.push_back(VertexNT(vec3(-1, t, 0), vec3(), vec2()));
    mesh->vertices.push_back(VertexNT(vec3(1, t, 0), vec3(), vec2()));
    mesh->vertices.push_back(VertexNT(vec3(-1, -t, 0), vec3(), vec2()));
    mesh->vertices.push_back(VertexNT(vec3(1, -t, 0), vec3(), vec2()));

    mesh->vertices.push_back(VertexNT(vec3(0, -1, t), vec3(), vec2()));
    mesh->vertices.push_back(VertexNT(vec3(0, 1, t), vec3(), vec2()));
    mesh->vertices.push_back(VertexNT(vec3(0, -1, -t), vec3(), vec2()));
    mesh->vertices.push_back(VertexNT(vec3(0, 1, -t), vec3(), vec2()));

    mesh->vertices.push_back(VertexNT(vec3(t, 0, -1), vec3(), vec2()));
    mesh->vertices.push_back(VertexNT(vec3(t, 0, 1), vec3(), vec2()));
    mesh->vertices.push_back(VertexNT(vec3(-t, 0, -1), vec3(), vec2()));
    mesh->vertices.push_back(VertexNT(vec3(-t, 0, 1), vec3(), vec2()));

    for (VertexNT& v : mesh->vertices)
    {
        v.position = make_vec4(normalize(make_vec3(v.position)), 1);
        v.normal   = v.position;
    }

    mesh->faces.push_back(Face(0, 11, 5));
    mesh->faces.push_back(Face(0, 5, 1));
    mesh->faces.push_back(Face(0, 1, 7));
    mesh->faces.push_back(Face(0, 7, 10));
    mesh->faces.push_back(Face(0, 10, 11));

    mesh->faces.push_back(Face(1, 5, 9));
    mesh->faces.push_back(Face(5, 11, 4));
    mesh->faces.push_back(Face(11, 10, 2));
    mesh->faces.push_back(Face(10, 7, 6));
    mesh->faces.push_back(Face(7, 1, 8));

    mesh->faces.push_back(Face(3, 9, 4));
    mesh->faces.push_back(Face(3, 4, 2));
    mesh->faces.push_back(Face(3, 2, 6));
    mesh->faces.push_back(Face(3, 6, 8));
    mesh->faces.push_back(Face(3, 8, 9));

    mesh->faces.push_back(Face(4, 9, 5));
    mesh->faces.push_back(Face(2, 4, 11));
    mesh->faces.push_back(Face(6, 2, 10));
    mesh->faces.push_back(Face(8, 6, 7));
    mesh->faces.push_back(Face(9, 8, 1));


    for (int r = 0; r < resolution; r++)
    {
        int faces = mesh->faces.size();
        for (int i = 0; i < faces; i++)
        {
            mesh->subdivideFace(i);
        }

        for (VertexNT& v : mesh->vertices)
        {
            v.position = make_vec4(normalize(make_vec3(v.position)), 1);
            v.normal   = v.position;
        }
    }

    mat4 S = scale(make_vec3(sphere.r));
    mat4 T = translate(sphere.pos);
    mesh->transform(T * S);

    return std::shared_ptr<default_mesh_t>(mesh);
}


std::shared_ptr<default_mesh_t> TriangleMeshGenerator::createCylinderMesh(float radius, float height, int sectors)
{
    default_mesh_t* mesh = new default_mesh_t();

    float const S = 1.f / (float)(sectors);

    for (int s = 0; s < sectors; s++)
    {
        float x = radius * cos(2 * M_PI * s * S);
        float y = -height / 2;
        float z = radius * sin(2 * M_PI * s * S);

        VertexNT vert(vec3(x, y, z), vec3(x, y, z));
        mesh->vertices.push_back(vert);
    }

    for (int s = 0; s < sectors; s++)
    {
        float x = radius * cos(2 * M_PI * s * S);
        float y = height / 2;
        float z = radius * sin(2 * M_PI * s * S);


        VertexNT vert(vec3(x, y, z), vec3(x, y, z));
        mesh->vertices.push_back(vert);
    }

    mesh->vertices.push_back(VertexNT(vec3(0, height / 2, 0), vec3(0, 1, 0), vec2()));
    mesh->vertices.push_back(VertexNT(vec3(0, -height / 2, 0), vec3(0, 1, 0), vec2()));

    for (uint32_t s = 0; s < uint32_t(sectors); s++)
    {
        //            uint32_t f[] = {s,(s+1)%sectors,sectors + (s+1)%sectors,sectors + (s)};
        uint32_t f[] = {s, sectors + (s), sectors + (s + 1) % sectors, (s + 1) % sectors};
        mesh->addQuad(f);

        {
            Face face;
            face.v3 = sectors + s;
            face.v2 = sectors + (s + 1) % sectors;
            face.v1 = 2 * sectors;
            mesh->faces.push_back(face);
        }
        {
            Face face;
            face.v1 = s;
            face.v2 = (s + 1) % sectors;
            face.v3 = 2 * sectors + 1;
            mesh->faces.push_back(face);
        }
    }

    return std::shared_ptr<default_mesh_t>(mesh);
}


std::shared_ptr<default_mesh_t> TriangleMeshGenerator::PlaneMesh(const Plane& plane)
{
    default_mesh_t* mesh = new default_mesh_t();
    mesh->vertices.push_back(VertexNT(vec3(-1, 0, -1), vec3(0, 1, 0), vec2(0, 0)));
    mesh->vertices.push_back(VertexNT(vec3(1, 0, -1), vec3(0, 1, 0), vec2(1, 0)));
    mesh->vertices.push_back(VertexNT(vec3(1, 0, 1), vec3(0, 1, 0), vec2(1, 1)));
    mesh->vertices.push_back(VertexNT(vec3(-1, 0, 1), vec3(0, 1, 0), vec2(0, 1)));

    mesh->faces.push_back(Face(0, 2, 1));
    mesh->faces.push_back(Face(0, 3, 2));
    return std::shared_ptr<default_mesh_t>(mesh);
}
#endif

std::shared_ptr<default_mesh_t> TriangleMeshGenerator::ConeMesh(const Cone& cone, int sectors)
{
    default_mesh_t* mesh = new default_mesh_t();
    mesh->vertices.push_back(VertexNT(vec3(0, 0, 0), vec3(0, 1, 0), vec2(0, 0)));              // top
    mesh->vertices.push_back(VertexNT(vec3(0, -cone.height, 0), vec3(0, -1, 0), vec2(0, 0)));  // bottom

    float const R = 1. / (float)(sectors);
    float const r = cone.radius;  // radius

    for (int s = 0; s < sectors; s++)
    {
        float x = r * sin((float)s * R * M_PI * 2.0f);
        float y = r * cos((float)s * R * M_PI * 2.0f);
        mesh->vertices.push_back(VertexNT(vec3(x, -cone.height, y), normalize(vec3(x, 0, y)), vec2(0, 0)));
    }

    for (int s = 0; s < sectors; s++)
    {
        Face face;
        face.v1 = s + 2;
        face.v2 = ((s + 1) % sectors) + 2;
        face.v3 = 0;
        mesh->faces.push_back(face);

        face.v1 = 1;
        face.v2 = ((s + 1) % sectors) + 2;
        face.v3 = s + 2;
        mesh->faces.push_back(face);
    }

    return std::shared_ptr<default_mesh_t>(mesh);
}


#if 0
std::shared_ptr<default_mesh_t> TriangleMeshGenerator::BoxMesh(const AABB& box)
{
    default_mesh_t* mesh = new default_mesh_t();

    unsigned int indices[]{
        0, 1, 2, 3,  // left
        7, 6, 5, 4,  // right
        1, 0, 4, 5,  // bottom
        3, 2, 6, 7,  // top
        0, 3, 7, 4,  // back
        2, 1, 5, 6   // front
    };


    // default
    //    vec2 texCoords[]{
    //        {0,0},{0,1},{1,1},{1,0},
    //        {0,0},{0,1},{1,1},{1,0},
    //        {0,0},{0,1},{1,1},{1,0},
    //        {0,0},{0,1},{1,1},{1,0},
    //        {0,0},{0,1},{1,1},{1,0},
    //        {0,0},{0,1},{1,1},{1,0}
    //    };

    // cube strip
#    define CUBE_EPSILON 0.0001f
    vec2 texCoords[]{{1.0f / 6.0f, 0.0f},
                     {0.0f / 6.0f, 0.0f},
                     {0.0f / 6.0f, 1.0f},
                     {1.0f / 6.0f, 1.0f},
                     {2.0f / 6.0f, 1.0f},
                     {3.0f / 6.0f, 1.0f},
                     {3.0f / 6.0f, 0.0f},
                     {2.0f / 6.0f, 0.0f},
                     {5.0f / 6.0f + CUBE_EPSILON, 0.0f},
                     {5.0f / 6.0f + CUBE_EPSILON, 1.0f},
                     {6.0f / 6.0f, 1.0f},
                     {6.0f / 6.0f, 0.0f},  // bottom
                     {5.0f / 6.0f - CUBE_EPSILON, 0.0f},
                     {4.0f / 6.0f + CUBE_EPSILON, 0.0f},
                     {4.0f / 6.0f + CUBE_EPSILON, 1.0f},
                     {5.0f / 6.0f - CUBE_EPSILON, 1.0f},  // top
                     {1.0f / 6.0f, 0.0f},
                     {1.0f / 6.0f, 1.0f},
                     {2.0f / 6.0f, 1.0f},
                     {2.0f / 6.0f, 0.0f},
                     {4.0f / 6.0f - CUBE_EPSILON, 1.0f},
                     {4.0f / 6.0f - CUBE_EPSILON, 0.0f},
                     {3.0f / 6.0f, 0.0f},
                     {3.0f / 6.0f, 1.0f}

    };

    vec3 normals[]{{-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}};


    for (int i = 0; i < 6; i++)
    {
        VertexNT verts[] = {VertexNT(box.cornerPoint(indices[i * 4 + 0]), normals[i], texCoords[i * 4 + 0]),
                            VertexNT(box.cornerPoint(indices[i * 4 + 1]), normals[i], texCoords[i * 4 + 1]),
                            VertexNT(box.cornerPoint(indices[i * 4 + 2]), normals[i], texCoords[i * 4 + 2]),
                            VertexNT(box.cornerPoint(indices[i * 4 + 3]), normals[i], texCoords[i * 4 + 3])};

        mesh->addQuad(verts);
    }

    return std::shared_ptr<default_mesh_t>(mesh);
}

std::shared_ptr<default_mesh_t> TriangleMeshGenerator::createSkyboxMesh(const AABB& box)
{
    std::shared_ptr<default_mesh_t> mesh = BoxMesh(box);


    for (unsigned int i = 0; i < mesh->faces.size(); i++)
    {
        mesh->invertFace(i);
    }
    return mesh;
}
#endif


}  // namespace Saiga
