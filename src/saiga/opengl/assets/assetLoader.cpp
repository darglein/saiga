/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "assetLoader.h"

#include "saiga/core/geometry/triangle_mesh_generator.h"
#include "saiga/core/image/imageGenerator.h"
#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/opengl/texture/TextureLoader.h"

namespace Saiga
{
AssetLoader::AssetLoader() {}

AssetLoader::~AssetLoader() {}



std::shared_ptr<ColoredAsset> AssetLoader::loadDebugArrow(float radius, float length, vec4 color)
{
    //    auto plainMesh = TriangleMeshGenerator::createMesh(Plane());
    //    auto cylinderMesh = TriangleMeshGenerator::createCylinderMesh(radius, length, 12);
    auto cylinderMesh = CylinderMesh(radius, length, 12).Mesh<VertexNC, uint32_t>();

    mat4 m = translate(vec3(0, length * 0.5f, 0));
    cylinderMesh.transform(m);

    float coneH   = length * 0.3f;
    float coneR   = radius * 1.3f;
    auto coneMesh = TriangleMeshGenerator::ConeMesh(Cone(make_vec3(0), vec3(0, 1, 0), coneR, coneH), 12);
    m             = translate(vec3(0, length + coneH, 0));
    coneMesh->transform(m);

    auto asset = std::make_shared<ColoredAsset>();
    asset->addMesh(cylinderMesh);
    asset->addMesh(*coneMesh);

    for (auto& v : asset->vertices)
    {
        v.color = color;
        v.data  = vec4(0.5, 0, 0, 0);
    }

    asset->create();
    return asset;
}


std::shared_ptr<ColoredAsset> AssetLoader::assetFromMesh(TriangleMesh<VertexNT, GLuint>& mesh, const vec4& color)
{
    auto asset = std::make_shared<ColoredAsset>();
    asset->addMesh(mesh);

    for (auto& v : asset->vertices)
    {
        v.color = color;
        v.data  = vec4(0.5, 0, 0, 0);
    }

    asset->create();
    return asset;
}

std::shared_ptr<ColoredAsset> AssetLoader::assetFromMesh(TriangleMesh<VertexNC, GLuint>& mesh)
{
    auto asset = std::make_shared<ColoredAsset>();
    asset->addMesh(mesh);

    for (auto& v : asset->vertices)
    {
        v.data = vec4(0.5, 0, 0, 0);
    }

    asset->create();
    return asset;
}

std::shared_ptr<ColoredAsset> AssetLoader::nonTriangleMesh(std::vector<vec3> vertices, std::vector<GLuint> indices,
                                                           GLenum mode, const vec4& color)
{
    std::shared_ptr<ColoredAsset> asset = std::make_shared<ColoredAsset>();

    for (auto v : vertices)
    {
        asset->vertices.push_back(VertexNC(v, vec3(0, 1, 0), make_vec3(color)));
    }
    //    for(auto& v : asset->vertices){
    //        v.color = color;
    //        v.data = vec4(0.5,0,0,0);
    //    }
    asset->loadDefaultShaders();

    asset->buffer.set(asset->vertices, indices, GL_STATIC_DRAW);
    asset->buffer.setDrawMode(mode);
    return asset;
}



static void createFrustumMesh(mat4 proj, std::vector<vec3>& vertices, std::vector<GLuint>& indices)
{
    float d = 1.0f;
    vec4 bl(-1, -1, d, 1);
    vec4 br(1, -1, d, 1);
    vec4 tl(-1, 1, d, 1);
    vec4 tr(1, 1, d, 1);

    mat4 projInv = inverse(proj);
    tl           = projInv * tl;
    tr           = projInv * tr;
    bl           = projInv * bl;
    br           = projInv * br;

    tl /= tl[3];
    tr /= tr[3];
    bl /= bl[3];
    br /= br[3];


    //    std::vector<VertexNC> vertices;

    vec4 positions[] = {
        vec4(0, 0, 0, 1), tl, tr, br, bl,
        //        vec4(tl[0],tl[1],-1,1),
        //        vec4(tr[0],tr[1],-1,1),
        //        vec4(br[0],br[1],-1,1),
        //        vec4(bl[0],bl[1],-1,1),


        0.4f * tl + 0.6f * tr, 0.6f * tl + 0.4f * tr, 0.5f * tl + 0.5f * tr + vec4(0, (tl[1] - bl[1]) * 0.1f, 0, 0)
        //        vec4(0.6*tl[0]+0.4*tr[0],0.6*tl[1]+0.4*tr[1],-1,1),
        //        vec4(0.5*tl[0]+0.5*tr[0],0.5*tl[1]+0.5*tr[1]-0.1,-1,1),

    };

    for (int i = 0; i < 8; ++i)
    {
        //        Vertex v;
        //        v.position = positions[i];
        vertices.push_back(make_vec3(positions[i]));
    }


    std::vector<GLuint> indices2 = {0, 1, 0, 2, 0, 3, 0, 4,

                                    1, 2, 3, 4, 1, 4, 2, 3,

                                    5, 7, 6, 7};
    indices                      = indices2;
}

std::shared_ptr<ColoredAsset> AssetLoader::frustumMesh(const mat4& proj, const vec4& color)
{
    std::vector<vec3> vertices;
    std::vector<GLuint> indices;
    createFrustumMesh(proj, vertices, indices);
    return nonTriangleMesh(vertices, indices, GL_LINES, color);
}

}  // namespace Saiga
