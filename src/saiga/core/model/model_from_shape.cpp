/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "model_from_shape.h"

#include "saiga/core/math/CoordinateSystems.h"

namespace Saiga
{
UnifiedMesh FullScreenQuad()
{
    UnifiedMesh model;

    model.position            = {vec3(-1, -1, 0), vec3(1, -1, 0), vec3(1, 1, 0), vec3(-1, 1, 0)};
    model.normal              = {vec3(0, 0, 1), vec3(0, 0, 1), vec3(0, 0, 1), vec3(0, 0, 1)};
    model.texture_coordinates = {vec2(0, 0), vec2(1, 0), vec2(1, 1), vec2(0, 1)};

    model.triangles = {ivec3(0, 1, 2), ivec3(0, 2, 3)};

    return model;
}

UnifiedMesh UVSphereMesh(const Sphere& sphere, int rings, int sectors)
{
    UnifiedMesh model;
    float const R = 1.f / (float)(rings);
    float const S = 1.f / (float)(sectors);

    for (int r = 0; r < rings + 1; r++)
    {
        for (int s = 0; s < sectors; s++)
        {
            float y = sphere.r * sin(-two_pi<float>() + pi<float>() * r * R);
            float x = sphere.r * cos(2 * pi<float>() * s * S) * sin(pi<float>() * r * R);
            float z = sphere.r * sin(2 * pi<float>() * s * S) * sin(pi<float>() * r * R);

            model.position.push_back(vec3(x, y, z).normalized());
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

UnifiedMesh IcoSphereMesh(const Sphere& sphere, int resolution)
{
    UnifiedMesh model;

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

UnifiedMesh CylinderMesh(float radius, float height, int sectors)
{
    UnifiedMesh model;

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

UnifiedMesh ConeMesh(const Cone& cone, int sectors)
{
    float const R = 1. / (float)(sectors);
    float const r = cone.radius;  // radius

    UnifiedMesh model;


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

    quat q = quat::FromTwoVectors(vec3(0, -1, 0), cone.direction);

    model.transform(make_mat4(q));

    return model;
}

UnifiedMesh PlaneMesh(const Plane& plane)
{
    UnifiedMesh model;
    model.position            = {vec3(-1, 0, -1), vec3(1, 0, -1), vec3(1, 0, 1), vec3(-1, 0, 1)};
    model.normal              = {vec3(0, 1, 0), vec3(0, 1, 0), vec3(0, 1, 0), vec3(0, 1, 0)};
    model.texture_coordinates = {vec2(0, 0), vec2(1, 0), vec2(1, 1), vec2(0, 1)};
    model.triangles           = {ivec3(0, 2, 1), ivec3(0, 3, 2)};
    return model;
}

UnifiedMesh BoxMesh(const AABB& box)
{
    UnifiedMesh model;


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

UnifiedMesh SkyboxMesh(const AABB& box)
{
    UnifiedMesh model = BoxMesh(box);
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

UnifiedMesh CheckerBoardPlane(const ivec2& size, float quadSize, const vec4& color1, const vec4& color2)
{
    UnifiedMesh model;
    vec4 n(0, 1, 0, 0);
    for (int i = -size[0]; i < size[0]; ++i)
    {
        for (int j = -size[1]; j < size[1]; ++j)
        {
            vec4 c            = (j + i % 2) % 2 == 0 ? color1 : color2;
            VertexNC verts[4] = {
                {{(float)i, 0.f, (float)j, 1.f}, n, c},
                {{(float)i, 0.f, j + 1.f, 1.f}, n, c},
                {{(float)i + 1.f, 0.f, j + 1.f, 1.f}, n, c},
                {{(float)i + 1.f, 0.f, (float)j, 1.f}, n, c},
            };

            int v_size = model.position.size();
            for (int i = 0; i < 4; ++i)
            {
                verts[i].position[0] *= quadSize;
                verts[i].position[2] *= quadSize;

                model.position.push_back(verts[i].position.head<3>());
                model.normal.push_back(verts[i].normal.head<3>());
                model.color.push_back(verts[i].color);
            }


            model.triangles.push_back(ivec3(v_size, v_size + 1, v_size + 2));
            model.triangles.push_back(ivec3(v_size, v_size + 2, v_size + 3));
        }
    }
    return model;
}

// ============= Line Meshes =============

UnifiedMesh GridBoxLineMesh(const AABB& box, ivec3 vsteps)
{
    UnifiedMesh model;

    for (int dim = 0; dim < 3; ++dim)
    {
        int steps = vsteps(dim);
        for (int j = 1; j <= 2; ++j)
        {
            int next_dim = (dim + j) % 3;
            for (int i = 0; i <= steps; ++i)
            {
                float alpha = i / float(steps);
                float x     = box.min(dim) + (box.max(dim) - box.min(dim)) * alpha;

                {
                    vec3 p1      = box.min;
                    vec3 p2      = box.max;
                    p1(dim)      = x;
                    p2(dim)      = x;
                    p1(next_dim) = p2(next_dim);

                    int id = model.NumVertices();
                    model.position.push_back(p1);
                    model.position.push_back(p2);

                    vec3 n(0, 0, 0);
                    n(next_dim) = -1;
                    model.normal.push_back(n);
                    model.normal.push_back(n);

                    model.lines.push_back({id, id + 1});
                }
                if (1)
                {
                    vec3 p1      = box.min;
                    vec3 p2      = box.max;
                    p1(dim)      = x;
                    p2(dim)      = x;
                    p2(next_dim) = p1(next_dim);

                    int id = model.NumVertices();
                    model.position.push_back(p1);
                    model.position.push_back(p2);

                    vec3 n(0, 0, 0);
                    n(next_dim) = 1;
                    model.normal.push_back(n);
                    model.normal.push_back(n);

                    model.lines.push_back({id, id + 1});
                }
            }
        }
    }
    return model;
}

UnifiedMesh GridPlaneLineMesh(const ivec2& dimension, const vec2& spacing)
{
    UnifiedMesh model;
    vec2 size = dimension.cast<float>().array() * spacing.array();


    std::vector<vec3> vertices;

    for (float i = -dimension.x(); i <= dimension.x(); i++)
    {
        vec3 p1 = vec3(spacing.x() * i, 0, -size[1]);
        vec3 p2 = vec3(spacing.x() * i, 0, size[1]);
        model.lines.push_back({vertices.size(), vertices.size() + 1});
        vertices.push_back(p1);
        vertices.push_back(p2);
    }

    for (float i = -dimension.y(); i <= dimension.y(); i++)
    {
        vec3 p1 = vec3(-size[0], 0, spacing.y() * i);
        vec3 p2 = vec3(+size[0], 0, spacing.y() * i);
        model.lines.push_back({vertices.size(), vertices.size() + 1});
        vertices.push_back(p1);
        vertices.push_back(p2);
    }



    for (auto v : vertices)
    {
        model.position.push_back(v);
    }
    return model;
}

UnifiedMesh FrustumLineMesh(const mat4& proj, float farPlaneDistance, bool vulkanTransform)
{
    UnifiedMesh model;

    float d = 1.0f;
    vec4 bl(-1, -1, d, 1);
    vec4 br(1, -1, d, 1);
    vec4 tl(-1, 1, d, 1);
    vec4 tr(1, 1, d, 1);

    mat4 tmp     = (inverse(GL2VulkanNormalizedImage()) * proj);
    mat4 projInv = vulkanTransform ? inverse(tmp) : inverse(proj);



    tl = projInv * tl;
    tr = projInv * tr;
    bl = projInv * bl;
    br = projInv * br;

    tl /= tl[3];
    tr /= tr[3];
    bl /= bl[3];
    br /= br[3];

    if (farPlaneDistance > 0)
    {
        tl[3] = -tl[2] / farPlaneDistance;
        tr[3] = -tr[2] / farPlaneDistance;
        bl[3] = -bl[2] / farPlaneDistance;
        br[3] = -br[2] / farPlaneDistance;

        tl /= tl[3];
        tr /= tr[3];
        bl /= bl[3];
        br /= br[3];
    }


    //    std::vector<VertexNC> vertices;

    vec4 positions[] = {vec4(0, 0, 0, 1),
                        tl,
                        tr,
                        br,
                        bl,
                        0.4f * tl + 0.6f * tr,
                        0.6f * tl + 0.4f * tr,
                        0.5f * tl + 0.5f * tr + vec4(0, (tl[1] - bl[1]) * 0.1f, 0, 0)};

    for (int i = 0; i < 8; ++i)
    {
        vec4 v = positions[i];

        model.position.push_back(v.head<3>());
    }


    model.lines = {{0, 1}, {0, 2}, {0, 3}, {0, 4},

                   {1, 2}, {3, 4}, {1, 4}, {2, 3},

                   {5, 7}, {6, 7}};
    return model;
}

UnifiedMesh FrustumCVLineMesh(const mat3& K, float farPlaneDistance, int w, int h)
{
    UnifiedMesh model;
    vec3 bl(0, h, 1);
    vec3 br(w, h, 1);
    vec3 tl(0, 0, 1);
    vec3 tr(w, 0, 1);

    mat3 projInv = inverse(K);

    tl = projInv * tl;
    tr = projInv * tr;
    bl = projInv * bl;
    br = projInv * br;


    if (farPlaneDistance > 0)
    {
        tl *= farPlaneDistance;
        tr *= farPlaneDistance;
        bl *= farPlaneDistance;
        br *= farPlaneDistance;
    }

    vec3 positions[] = {vec3(0, 0, 0),
                        tl,
                        tr,
                        br,
                        bl,
                        0.4f * tl + 0.6f * tr,
                        0.6f * tl + 0.4f * tr,
                        0.5f * tl + 0.5f * tr + vec3(0, (tl[1] - bl[1]) * 0.1f, 0)};

    for (int i = 0; i < 8; ++i)
    {
        vec3 v = positions[i];
        model.position.push_back(v);
    }

    model.lines = {{0, 1}, {0, 2}, {0, 3}, {0, 4},

                   {1, 2}, {3, 4}, {1, 4}, {2, 3},

                   {5, 7}, {6, 7}};
    return model;
}

UnifiedMesh FrustumLineMesh(const Frustum& frustum)
{
    UnifiedMesh model;

    auto tris = frustum.ToTriangleList();
    for (auto tri : tris)
    {
        int id = model.NumVertices();

        model.position.push_back(tri.a);
        model.position.push_back(tri.b);
        model.position.push_back(tri.c);

        model.lines.push_back({id + 0, id + 1});
        model.lines.push_back({id + 1, id + 2});
    }
    return model;
}
UnifiedMesh SimpleHeightmap(const ImageView<uint16_t> image, float height_scale, float horizontal_scale,
                            bool translate_to_origin)
{
    UnifiedMesh model;

    int h = image.h;
    int w = image.w;

    float inv_h = 1.0f / (h);
    float inv_w = 1.0f / (w);

    std::cout << "hm " << inv_h << " " << inv_w << std::endl;

    SAIGA_ASSERT(h == w);


    vec2 horizontal_translation = vec2::Zero();

    if (translate_to_origin)
    {
        horizontal_translation = vec2(1, 1) * horizontal_scale * -0.5;
    }

    // Create vertices
    for (int i = 0; i < h; ++i)
    {
        for (int j = 0; j < w; ++j)
        {
            float h = image(i, j) * height_scale * (1.0f / std::numeric_limits<uint16_t>::max());

            vec2 xz = vec2(j, i).array() * vec2(inv_w, inv_h).array();

            xz = xz * horizontal_scale + horizontal_translation;


            vec3 p(xz(0), h, xz(1));
            model.position.push_back(p);
            model.normal.push_back(vec3(0, 1, 0));
            model.color.push_back(vec4(1, 1, 1, 1));
        }
    }

    // Triangles
    for (int i = 0; i < h - 1; ++i)
    {
        for (int j = 0; j < w - 1; ++j)
        {
            int p1 = i * w + j;
            int p2 = i * w + j + 1;
            int p3 = (i + 1) * w + j;
            int p4 = (i + 1) * w + j + 1;
            model.triangles.push_back(ivec3(p1, p3, p2));
            model.triangles.push_back(ivec3(p2, p3, p4));
        }
    }


    model.CalculateVertexNormals();


    return model;
}
UnifiedMesh CoordinateSystemMesh(float scale, bool add_sphere)
{
    UnifiedMesh result;

    auto base_cylinder = CylinderMesh(0.05, 1, 10).transform(translate(vec3(0, 0.5, 0)));

    result = UnifiedMesh(base_cylinder).SetVertexColor(vec4(0, 1, 0, 1));
    result = UnifiedMesh(result, UnifiedMesh(base_cylinder)
                                     .SetVertexColor(vec4(0, 0, 1, 1))
                                     .transform(Saiga::rotate(radians(90.f), vec3(1, 0, 0))));
    result = UnifiedMesh(result, UnifiedMesh(base_cylinder)
                                     .SetVertexColor(vec4(1, 0, 0, 1))
                                     .transform(Saiga::rotate(radians(-90.f), vec3(0, 0, 1))));

    if (add_sphere)
    {
        result = UnifiedMesh(result, IcoSphereMesh(Sphere(vec3(0, 0, 0), 0.1), 3).SetVertexColor(vec4(1, 1, 1, 1)));
    }

    result.transform(Saiga::scale(vec3(scale, scale, scale)));

    return result;
}


}  // namespace Saiga
