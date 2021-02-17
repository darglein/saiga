/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "triangle_mesh_generator2.h"

#include "internal/noGraphicsAPI.h"
namespace Saiga
{
UnifiedModel FullScreenQuad()
{
    UnifiedModel model;

    model.position            = {vec3(-1, -1, 0), vec3(1, -1, 0), vec3(1, 1, 0), vec3(-1, 1, 0)};
    model.normal              = {vec3(0, 0, 1), vec3(0, 0, 1), vec3(0, 0, 1), vec3(0, 0, 1)};
    model.texture_coordinates = {vec2(0, 0), vec2(1, 0), vec2(1, 1), vec2(0, 1)};

    model.triangles = {ivec3(0, 1, 2), ivec3(0, 2, 3)};

    return model;
}

UnifiedModel UVSphereMesh(const Sphere& sphere, int rings, int sectors)
{
    UnifiedModel model;
    float const R = 1.f / (float)(rings);
    float const S = 1.f / (float)(sectors);

    for (int r = 0; r < rings + 1; r++)
    {
        for (int s = 0; s < sectors; s++)
        {
            float y = sphere.r * sin(-two_pi<float>() + pi<float>() * r * R);
            float x = sphere.r * cos(2 * pi<float>() * s * S) * sin(pi<float>() * r * R);
            float z = sphere.r * sin(2 * pi<float>() * s * S) * sin(pi<float>() * r * R);

            model.position.push_back(vec3(x, y, z));
            model.normal.push_back(vec3(x, y, z).normalized());
            model.texture_coordinates.push_back(vec2(s * S, r * R));
        }
    }

    for (int r = 0; r < rings; r++)
    {
        for (int s = 0; s < sectors; s++)
        {
            if (r != rings - 1)
            {
                ivec3 f;
                f(0) = (r + 1) * sectors + s;
                f(1) = (r + 1) * sectors + (s + 1) % sectors;
                f(2) = r * sectors + (s + 1) % sectors;
                model.triangles.push_back(f);
            }
            if (r != 0)
            {
                ivec3 f;
                f(0) = (r + 1) * sectors + s;
                f(1) = r * sectors + (s + 1) % sectors;
                f(2) = r * sectors + s;
                model.triangles.push_back(f);
            }
        }
    }

    mat4 T = translate(sphere.pos);
    model.transform(T * S);

    return model;
}

UnifiedModel IcoSphereMesh(const Sphere& sphere, int resolution)
{
    UnifiedModel model;

    float t = (1.0 + sqrt(5.0)) / 2.0;

    model.position = {vec3(-1, t, 0), vec3(1, t, 0), vec3(-1, -t, 0), vec3(1, -t, 0),
                      vec3(0, -1, t), vec3(0, 1, t), vec3(0, -1, -t), vec3(0, 1, -t),
                      vec3(t, 0, -1), vec3(t, 0, 1), vec3(-t, 0, -1), vec3(-t, 0, 1)};

    model.triangles = {ivec3(0, 11, 5), ivec3(0, 5, 1),  ivec3(0, 1, 7),   ivec3(0, 7, 10), ivec3(0, 10, 11),
                       ivec3(1, 5, 9),  ivec3(5, 11, 4), ivec3(11, 10, 2), ivec3(10, 7, 6), ivec3(7, 1, 8),
                       ivec3(3, 9, 4),  ivec3(3, 4, 2),  ivec3(3, 2, 6),   ivec3(3, 6, 8),  ivec3(3, 8, 9),
                       ivec3(4, 9, 5),  ivec3(2, 4, 11), ivec3(6, 2, 10),  ivec3(8, 6, 7),  ivec3(9, 8, 1)};

    for (auto& p : model.position)
    {
        p.normalize();
    }

    // subdivide sphere according to the resolution
    for (int r = 0; r < resolution; ++r)
    {
        int faces = model.NumFaces();
        for (int i = 0; i < faces; i++)
        {
            auto face = model.triangles[i];

            auto p1 = model.position[face(0)];
            auto p2 = model.position[face(1)];
            auto p3 = model.position[face(2)];

            int v1 = model.position.size();
            int v2 = v1 + 1;
            int v3 = v1 + 2;
            model.position.push_back(((p1 + p2) * 0.5f).normalized());
            model.position.push_back(((p1 + p3) * 0.5f).normalized());
            model.position.push_back(((p2 + p3) * 0.5f).normalized());

            model.triangles.push_back(ivec3(face(1), v3, v1));
            model.triangles.push_back(ivec3(face(2), v2, v3));
            model.triangles.push_back(ivec3(v1, v3, v2));
            model.triangles[i] = ivec3(face(0), v1, v2);
        }
    }

    for (auto& p : model.position)
    {
        model.normal.push_back(p);
    }


    mat4 S = scale(make_vec3(sphere.r));
    mat4 T = translate(sphere.pos);
    model.transform(T * S);

    return model;
}

UnifiedModel CylinderMesh(float radius, float height, int sectors)
{
    UnifiedModel model;

    float const S = 1.f / (float)(sectors);

    for (int s = 0; s < sectors; s++)
    {
        float x = radius * cos(2 * pi<float>() * s * S);
        float y = -height / 2;
        float z = radius * sin(2 * pi<float>() * s * S);

        model.position.push_back(vec3(x, y, z));
    }

    for (int s = 0; s < sectors; s++)
    {
        float x = radius * cos(2 * pi<float>() * s * S);
        float y = height / 2;
        float z = radius * sin(2 * pi<float>() * s * S);

        model.position.push_back(vec3(x, y, z));
    }
    model.position.push_back(vec3(0, height / 2, 0));
    model.position.push_back(vec3(0, -height / 2, 0));

    for (uint32_t s = 0; s < uint32_t(sectors); s++)
    {
        //            uint32_t f[] = {s,(s+1)%sectors,sectors + (s+1)%sectors,sectors + (s)};
        //        uint32_t f[] = {s, sectors + (s), sectors + (s + 1) % sectors, (s + 1) % sectors};
        //        mesh->addQuad(f);

        model.triangles.push_back(ivec3(s, sectors + (s), sectors + (s + 1) % sectors));
        model.triangles.push_back(ivec3(sectors + (s + 1) % sectors, (s + 1) % sectors, s));

        {
            ivec3 face;
            face(2) = sectors + s;
            face(1) = sectors + (s + 1) % sectors;
            face(0) = 2 * sectors;
            model.triangles.push_back(face);
        }
        {
            ivec3 face;
            face(0) = s;
            face(1) = (s + 1) % sectors;
            face(2) = 2 * sectors + 1;
            model.triangles.push_back(face);
        }
    }

    for (auto& p : model.position)
    {
        model.normal.push_back(p.normalized());
    }



    return model;
}

UnifiedModel ConeMesh(const Cone& cone, int sectors)
{
    float const R = 1. / (float)(sectors);
    float const r = cone.radius;  // radius

    UnifiedModel model;


    model.position.push_back(vec3(0, 0, 0));             // top
    model.position.push_back(vec3(0, -cone.height, 0));  // bottom

    for (int s = 0; s < sectors; s++)
    {
        float x = r * sin((float)s * R * pi<float>() * 2.0f);
        float y = r * cos((float)s * R * pi<float>() * 2.0f);
        model.position.push_back(vec3(x, -cone.height, y));
    }

    for (int s = 0; s < sectors; s++)
    {
        ivec3 face;
        face(0) = s + 2;
        face(1) = ((s + 1) % sectors) + 2;
        face(2) = 0;
        model.triangles.push_back(face);

        face(0) = 1;
        face(1) = ((s + 1) % sectors) + 2;
        face(2) = s + 2;
        model.triangles.push_back(face);
    }

    return model;
}

UnifiedModel PlaneMesh(const Plane& plane)
{
    UnifiedModel model;
    model.position            = {vec3(-1, 0, -1), vec3(1, 0, -1), vec3(1, 0, 1), vec3(-1, 0, 1)};
    model.normal              = {vec3(0, 1, 0), vec3(0, 1, 0), vec3(0, 1, 0), vec3(0, 1, 0)};
    model.texture_coordinates = {vec2(0, 0), vec2(1, 0), vec2(1, 1), vec2(0, 1)};
    model.triangles           = {ivec3(0, 2, 1), ivec3(0, 3, 2)};
    return model;
}

UnifiedModel BoxMesh(const AABB& box)
{
    UnifiedModel model;


    unsigned int indices[]{
        0, 1, 2, 3,  // left
        7, 6, 5, 4,  // right
        1, 0, 4, 5,  // bottom
        3, 2, 6, 7,  // top
        0, 3, 7, 4,  // back
        2, 1, 5, 6   // front
    };


    // cube strip
    const float CUBE_EPSILON = 0.0001f;
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

        int z = model.position.size();

        for (int j = 0; j < 4; ++j)
        {
            model.position.push_back(verts[j].position.head<3>());
            model.texture_coordinates.push_back(verts[j].texture);
            model.normal.push_back(normals[i]);
        }


        model.triangles.push_back(ivec3(z, z + 1, z + 2));
        model.triangles.push_back(ivec3(z + 2, z + 3, z));
    }


    return model;
}

UnifiedModel SkyboxMesh(const AABB& box)
{
    UnifiedModel model = BoxMesh(box);
    for (auto& t : model.triangles)
    {
        ivec3 f2;
        f2(0) = t(2);
        f2(1) = t(1);
        f2(2) = t(0);
        t     = f2;
    }
    return model;
}

}  // namespace Saiga
