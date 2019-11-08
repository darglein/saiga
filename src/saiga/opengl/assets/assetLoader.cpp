/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "assetLoader.h"

#include "saiga/core/geometry/grid.h"
#include "saiga/core/geometry/triangle_mesh_generator.h"
#include "saiga/core/image/imageGenerator.h"
#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/opengl/texture/TextureLoader.h"

namespace Saiga
{
AssetLoader::AssetLoader() {}

AssetLoader::~AssetLoader() {}


void AssetLoader::loadBasicShaders()
{
    if (basicAssetShader) return;
    basicAssetShader          = shaderLoader.load<MVPShader>("geometry/deferred_mvp_model.glsl");
    basicAssetForwardShader   = shaderLoader.load<MVPShader>("geometry/deferred_mvp_model_forward.glsl");
    basicAssetDepthshader     = shaderLoader.load<MVPShader>("geometry/deferred_mvp_model_depth.glsl");
    basicAssetWireframeShader = shaderLoader.load<MVPShader>("geometry/deferred_mvp_model_wireframe.glsl");
}

void AssetLoader::loadTextureShaders()
{
    if (texturedAssetShader) return;
    texturedAssetShader          = shaderLoader.load<MVPTextureShader>("geometry/texturedAsset.glsl");
    texturedAssetForwardShader   = shaderLoader.load<MVPTextureShader>("geometry/texturedAsset.glsl");
    texturedAssetDepthShader     = shaderLoader.load<MVPTextureShader>("geometry/texturedAsset_depth.glsl");
    texturedAssetWireframeShader = shaderLoader.load<MVPTextureShader>("geometry/texturedAsset_wireframe.glsl");
}

void AssetLoader::loadAnimatedShaders()
{
    if (animatedAssetShader) return;
    animatedAssetShader          = shaderLoader.load<BoneShader>("geometry/deferred_mvp_bones.glsl");
    animatedAssetDepthshader     = shaderLoader.load<BoneShader>("geometry/deferred_mvp_bones_depth.glsl");
    animatedAssetWireframeShader = shaderLoader.load<BoneShader>("geometry/deferred_mvp_bones.glsl");
}


std::shared_ptr<TexturedAsset> AssetLoader::loadDebugPlaneAsset(vec2 size, float quadSize, Color color1, Color color2)
{
    auto cbImage                       = ImageGenerator::checkerBoard(color1, color2, 16, 2, 2);
    std::shared_ptr<Texture> cbTexture = std::make_shared<Texture>();
    cbTexture->fromImage(*cbImage, true, true);
    cbTexture->setFiltering(GL_NEAREST);
    cbTexture->setWrap(GL_REPEAT);
    cbTexture->generateMipmaps();
    std::shared_ptr<TexturedAsset> asset = loadDebugTexturedPlane(cbTexture, size);
    for (auto& v : asset->vertices)
    {
        //        v.texture *= size / quadSize;
        v.texture = v.texture.array() * vec2(size * (1.0f / quadSize)).array();
    }
    //    asset->createBuffers(asset->buffer);
    asset->buffer.fromMesh(*asset);
    return asset;
}

std::shared_ptr<ColoredAsset> AssetLoader::loadDebugPlaneAsset2(ivec2 size, float quadSize, Color color1, Color color2)
{
    std::shared_ptr<ColoredAsset> asset = std::make_shared<ColoredAsset>();

    asset->createCheckerBoard(size, quadSize, color1, color2);

    loadBasicShaders();
    asset->create(basicAssetShader, basicAssetForwardShader, basicAssetDepthshader, basicAssetWireframeShader);

    return asset;
}

std::shared_ptr<TexturedAsset> AssetLoader::loadDebugTexturedPlane(std::shared_ptr<Texture> texture, vec2 size)
{
    auto plainMesh = TriangleMeshGenerator::createMesh(Plane());
    mat4 S         = scale(vec3(size[0], 1, size[1]));
    plainMesh->transform(S);

    auto asset = std::make_shared<TexturedAsset>();

    asset->addMesh(*plainMesh);

    for (auto& v : asset->vertices)
    {
        v.data = vec4(0.5, 0, 0, 0);
    }

    TexturedAsset::TextureGroup tg;
    tg.startIndex = 0;
    tg.indices    = plainMesh->numIndices();
    tg.texture    = texture;
    asset->groups.push_back(tg);
    loadTextureShaders();
    asset->create(texturedAssetShader, texturedAssetForwardShader, texturedAssetDepthShader,
                  texturedAssetWireframeShader);

    return asset;
}

std::shared_ptr<ColoredAsset> AssetLoader::loadDebugGrid(int numX, int numY, float quadSize, Color color)
{
    vec2 size = vec2(numX, numY) * quadSize;

    std::vector<vec3> vertices;
    std::vector<GLuint> indices;

    for (float i = -numX; i <= numX; i++)
    {
        vec3 p1 = vec3(quadSize * i, 0, -size[1]);
        vec3 p2 = vec3(quadSize * i, 0, size[1]);
        indices.push_back(vertices.size());
        vertices.push_back(p1);
        indices.push_back(vertices.size());
        vertices.push_back(p2);
    }

    for (float i = -numY; i <= numY; i++)
    {
        vec3 p1 = vec3(-size[0], 0, quadSize * i);
        vec3 p2 = vec3(+size[0], 0, quadSize * i);
        indices.push_back(vertices.size());
        vertices.push_back(p1);
        indices.push_back(vertices.size());
        vertices.push_back(p2);
    }


    return nonTriangleMesh(vertices, indices, GL_LINES, color);
}

std::shared_ptr<ColoredAsset> AssetLoader::loadDebugArrow(float radius, float length, vec4 color)
{
    //    auto plainMesh = TriangleMeshGenerator::createMesh(Plane());
    auto cylinderMesh = TriangleMeshGenerator::createCylinderMesh(radius, length, 12);
    mat4 m            = translate(vec3(0, length * 0.5f, 0));
    cylinderMesh->transform(m);

    float coneH   = length * 0.3f;
    float coneR   = radius * 1.3f;
    auto coneMesh = TriangleMeshGenerator::createMesh(Cone(make_vec3(0), vec3(0, 1, 0), coneR, coneH), 12);
    m             = translate(vec3(0, length + coneH, 0));
    coneMesh->transform(m);

    auto asset = std::make_shared<ColoredAsset>();
    asset->addMesh(*cylinderMesh);
    asset->addMesh(*coneMesh);

    for (auto& v : asset->vertices)
    {
        v.color = color;
        v.data  = vec4(0.5, 0, 0, 0);
    }

    loadBasicShaders();
    asset->create(basicAssetShader, basicAssetForwardShader, basicAssetDepthshader, basicAssetWireframeShader);
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

    loadBasicShaders();
    asset->create(basicAssetShader, basicAssetForwardShader, basicAssetDepthshader, basicAssetWireframeShader);
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

    loadBasicShaders();
    asset->create(basicAssetShader, basicAssetForwardShader, basicAssetDepthshader, basicAssetWireframeShader);
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
    loadBasicShaders();
    asset->create(basicAssetShader, basicAssetForwardShader, basicAssetDepthshader, basicAssetWireframeShader);
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
