/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/assets/assetLoader.h"

#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/opengl/texture/textureLoader.h"

#include "saiga/geometry/triangle_mesh_generator.h"
#include "saiga/image/imageGenerator.h"

namespace Saiga {

AssetLoader::AssetLoader()
{
     loadDefaultShaders();
}

AssetLoader::~AssetLoader()
{

}


void AssetLoader::loadDefaultShaders()
{
    basicAssetShader = ShaderLoader::instance()->load<MVPShader>("geometry/deferred_mvp_model.glsl");
    basicAssetForwardShader = ShaderLoader::instance()->load<MVPShader>("geometry/deferred_mvp_model_forward.glsl");
    basicAssetDepthshader = ShaderLoader::instance()->load<MVPShader>("geometry/deferred_mvp_model_depth.glsl");
    basicAssetWireframeShader = ShaderLoader::instance()->load<MVPShader>("geometry/deferred_mvp_model_wireframe.glsl");

    texturedAssetShader = ShaderLoader::instance()->load<MVPTextureShader>("geometry/texturedAsset.glsl");
    texturedAssetDepthShader = ShaderLoader::instance()->load<MVPTextureShader>("geometry/texturedAsset_depth.glsl");
    texturedAssetWireframeShader = ShaderLoader::instance()->load<MVPTextureShader>("geometry/texturedAsset_wireframe.glsl");

    animatedAssetShader = ShaderLoader::instance()->load<BoneShader>("geometry/deferred_mvp_bones.glsl");
    animatedAssetDepthshader = ShaderLoader::instance()->load<BoneShader>("geometry/deferred_mvp_bones_depth.glsl");
    animatedAssetWireframeShader = ShaderLoader::instance()->load<BoneShader>("geometry/deferred_mvp_bones.glsl");
}

std::shared_ptr<TexturedAsset> AssetLoader::loadDebugPlaneAsset(vec2 size, float quadSize, Color color1, Color color2)
{
        auto cbImage = ImageGenerator::checkerBoard(color1,color2,16,2,2);
        std::shared_ptr<Texture> cbTexture = std::make_shared<Texture>();
        cbTexture->fromImage(*cbImage);
        cbTexture->setFiltering(GL_NEAREST);
        cbTexture->setWrap(GL_REPEAT);
        cbTexture->generateMipmaps();
        std::shared_ptr<TexturedAsset> asset = loadDebugTexturedPlane(cbTexture,size);
        for(auto &v : asset->mesh.vertices){
            v.texture *= size / quadSize;
        }
        asset->mesh.createBuffers(asset->buffer);
        return asset;
}

std::shared_ptr<TexturedAsset> AssetLoader::loadDebugTexturedPlane(std::shared_ptr<Texture> texture, vec2 size)
{
    auto plainMesh = TriangleMeshGenerator::createMesh(Plane());
    mat4 scale = glm::scale(mat4(1),vec3(size.x,1,size.y));
    plainMesh->transform(scale);

    auto asset = std::make_shared<TexturedAsset>();

    asset->mesh.addMesh(*plainMesh);

    for(auto& v : asset->mesh.vertices){
        v.data = vec4(0.5,0,0,0);
    }

    TexturedAsset::TextureGroup tg;
    tg.startIndex = 0;
    tg.indices = plainMesh->numIndices();
    tg.texture = texture;
    asset->groups.push_back(tg);
    asset->create("Plane",texturedAssetShader,texturedAssetForwardShader,texturedAssetDepthShader,texturedAssetWireframeShader);

    return asset;
}

std::shared_ptr<ColoredAsset> AssetLoader::loadDebugArrow(float radius, float length, vec4 color)
{
//    auto plainMesh = TriangleMeshGenerator::createMesh(Plane());
    auto cylinderMesh = TriangleMeshGenerator::createCylinderMesh(radius,length,12);
    mat4 m = glm::translate(mat4(),vec3(0,length*0.5f,0));
    cylinderMesh->transform(m);

    float coneH = length * 0.3f;
    float coneR = radius * 1.3f;
    auto coneMesh = TriangleMeshGenerator::createMesh(Cone(vec3(0),vec3(0,1,0),coneR,coneH),12);
    m = glm::translate(mat4(),vec3(0,length+coneH,0));
        coneMesh->transform(m);

    auto asset = std::make_shared<ColoredAsset>();
    asset->mesh.addMesh(*cylinderMesh);
    asset->mesh.addMesh(*coneMesh);

    for(auto& v : asset->mesh.vertices){
        v.color = color;
        v.data = vec4(0.5,0,0,0);
    }

    asset->create("Arrow",basicAssetShader,basicAssetForwardShader,basicAssetDepthshader,basicAssetWireframeShader);
    return asset;
}


std::shared_ptr<ColoredAsset> AssetLoader::assetFromMesh(TriangleMesh<VertexNT, GLuint>  &mesh, const vec4 &color)
{
    auto asset = std::make_shared<ColoredAsset>();
    asset->mesh.addMesh(mesh);

    for(auto& v : asset->mesh.vertices){
        v.color = color;
        v.data = vec4(0.5,0,0,0);
    }

    asset->create("Fromsdfg",basicAssetShader,basicAssetForwardShader,basicAssetDepthshader,basicAssetWireframeShader);
    return asset;
}

std::shared_ptr<ColoredAsset> AssetLoader::assetFromMesh(TriangleMesh<VertexNC, GLuint> &mesh)
{
    auto asset = std::make_shared<ColoredAsset>();
    asset->mesh.addMesh(mesh);

    for(auto& v : asset->mesh.vertices){
        v.data = vec4(0.5,0,0,0);
    }

    asset->create("Fromsdfg",basicAssetShader,basicAssetForwardShader,basicAssetDepthshader,basicAssetWireframeShader);
    return asset;
}

std::shared_ptr<ColoredAsset> AssetLoader::nonTriangleMesh(std::vector<vec3> vertices, std::vector<GLuint> indices, GLenum mode, const vec4& color)
{
    std::shared_ptr<ColoredAsset> asset = std::make_shared<ColoredAsset>();

    for(auto v : vertices){
        asset->mesh.vertices.push_back(VertexNC(v,vec3(0,1,0),vec3(color)));
    }
//    for(auto& v : asset->mesh.vertices){
//        v.color = color;
//        v.data = vec4(0.5,0,0,0);
//    }
    asset->create("Fromsdfg",basicAssetShader,basicAssetForwardShader,basicAssetDepthshader,basicAssetWireframeShader);
    asset->buffer.set(asset->mesh.vertices,indices,GL_STATIC_DRAW);
    asset->buffer.setDrawMode(mode);
    return asset;
}




static void createFrustumMesh(mat4 proj, std::vector<vec3>& vertices,  std::vector<GLuint>& indices)
{

    float d = 0.5f;
    vec4 bl(-1,-1,d,1);
    vec4 br(1,-1,d,1);
    vec4 tl(-1,1,d,1);
    vec4 tr(1,1,d,1);

    mat4 projInv = inverse(proj);
    tl = projInv * tl;
    tr = projInv * tr;
    bl = projInv * bl;
    br = projInv * br;

    tl /= tl.w;
    tr /= tr.w;
    bl /= bl.w;
    br /= br.w;


    //    std::vector<VertexNC> vertices;

    vec4 positions[] = {
        vec4(0,0,0,1),
        tl,tr,br,bl,
        //        vec4(tl[0],tl[1],-1,1),
        //        vec4(tr[0],tr[1],-1,1),
        //        vec4(br[0],br[1],-1,1),
        //        vec4(bl[0],bl[1],-1,1),


        0.4f * tl + 0.6f * tr,
        0.6f * tl + 0.4f * tr,
        0.5f * tl + 0.5f * tr + vec4(0,(tl.y-bl.y)*0.1f,0,0)
        //        vec4(0.6*tl[0]+0.4*tr[0],0.6*tl[1]+0.4*tr[1],-1,1),
        //        vec4(0.5*tl[0]+0.5*tr[0],0.5*tl[1]+0.5*tr[1]-0.1,-1,1),

    };

    for(int i = 0 ; i < 8 ; ++i){
//        Vertex v;
//        v.position = positions[i];
        vertices.push_back(vec3(positions[i]));
    }


    std::vector<GLuint> indices2 = {
        0,1,
        0,2,
        0,3,
        0,4,

        1,2,
        3,4,
        1,4,
        2,3,

        5,7,
        6,7
    };
    indices = indices2;
}

std::shared_ptr<ColoredAsset> AssetLoader::frustumMesh(const mat4 &proj, const vec4 &color)
{
     std::vector<vec3> vertices;
     std::vector<GLuint> indices;
     createFrustumMesh(proj,vertices,indices);

     return nonTriangleMesh(vertices,indices,GL_LINES,color);

}

}
