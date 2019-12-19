/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "ModelVertexColored.h"

#include "saiga/core/geometry/triangle_mesh_generator.h"

#include "internal/noGraphicsAPI.h"

#include "objModelLoader.h"
#include "plyModelLoader.h"

namespace Saiga
{
void VertexColoredModel::createFullscreenQuad()
{
    vertices.push_back(VertexNC(vec3(-1, -1, 0), vec3(0, 0, 1)));
    vertices.push_back(VertexNC(vec3(1, -1, 0), vec3(0, 0, 1)));
    vertices.push_back(VertexNC(vec3(1, 1, 0), vec3(0, 0, 1)));
    vertices.push_back(VertexNC(vec3(-1, 1, 0), vec3(0, 0, 1)));
    addFace(0, 2, 3);
    addFace(0, 1, 2);
}


static TriangleMesh<VertexNC, uint32_t> ArrowMeshY(float radius, float length, const vec4& color)
{
    float coneH = length * 0.15f;
    float coneR = radius * 1.5f;


    auto cylinderMesh = TriangleMeshGenerator::createCylinderMesh(radius, length - coneH, 12);
    mat4 m            = translate(vec3(0, (length - coneH) * 0.5f, 0));
    cylinderMesh->transform(m);

    auto coneMesh = TriangleMeshGenerator::createMesh(Cone(make_vec3(0), vec3(0, 1, 0), coneR, coneH), 12);
    m             = translate(vec3(0, length, 0));
    coneMesh->transform(m);

    TriangleMesh<VertexNC, uint32_t> mesh;
    mesh.addMesh(*cylinderMesh);
    mesh.addMesh(*coneMesh);
    mesh.setColor(color);
    return mesh;
}

void VertexColoredModel::createArrow(float radius, float length, const vec4& color)
{
    addMesh(ArrowMeshY(radius, length, color));
}

void VertexColoredModel::createCoordinateSystem(float _scale, bool full)
{
    float radius = 0.05;

    float length = 1;
    if (full)
    {
        length = 2;
    }

    auto x = ArrowMeshY(radius, length, vec4(1, 0, 0, 1));
    x.transform(rotate(radians(90), vec3(0, 0, -1)));

    auto y = ArrowMeshY(radius, length, vec4(0, 1, 0, 1));

    auto z = ArrowMeshY(radius, length, vec4(0, 0, 1, 1));
    z.transform(rotate(radians(90), vec3(1, 0, 0)));

    if (full)
    {
        x.transform(translate(vec3(-1, 0, 0)));
        y.transform(translate(vec3(0, -1, 0)));
        z.transform(translate(vec3(0, 0, -1)));
    }

    addMesh(x);
    addMesh(y);
    addMesh(z);
    transform(scale(vec3(_scale, _scale, _scale)));
}

void VertexColoredModel::createCheckerBoard(ivec2 size, float quadSize, const vec4& color1, const vec4& color2)
{
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

            for (int i = 0; i < 4; ++i)
            {
                verts[i].position[0] *= quadSize;
                verts[i].position[2] *= quadSize;
            }

            addQuad(verts);
        }
    }
}

void VertexColoredModel::loadObj(const std::string& file)
{
    Saiga::ObjModelLoader loader(file);
    loader.computeVertexColorAndData();
    loader.toTriangleMesh(*this);
}

void VertexColoredModel::loadPly(const std::string& file)
{
    Saiga::PLYLoader loader(file);
    this->TriangleMesh<VertexNC, uint32_t>::operator=(loader.mesh);
}

void TexturedModel::loadObj(const std::string& file)
{
    Saiga::ObjModelLoader loader(file);
    loader.computeVertexColorAndData();
    loader.toTriangleMesh(*this);

    for (ObjTriangleGroup& otg : loader.triangleGroups)
    {
        if (otg.faces == 0) continue;

        TextureGroup tg;
        tg.indices    = otg.faces * 3;
        tg.startIndex = otg.startFace * 3;


        Material m;
        m.diffuse   = otg.material.map_Kd;
        tg.material = m;

        groups.push_back(tg);
    }
}


}  // namespace Saiga
