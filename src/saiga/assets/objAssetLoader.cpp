/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/assets/objAssetLoader.h"

#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/opengl/texture/textureLoader.h"
#include "saiga/assets/model/objModelLoader.h"

namespace Saiga {

ObjAssetLoader::ObjAssetLoader(){

}

ObjAssetLoader::~ObjAssetLoader()
{

}

void ObjAssetLoader::loadMeshNC(const std::string &file, TriangleMesh<VertexNC, GLuint> &mesh, bool normalize)
{
    mesh.vertices.clear();
    mesh.faces.clear();

    ObjLoader2 ol(file);
    TriangleMesh<VertexNC,GLuint> &tmesh = mesh;

    for(ObjTriangle &oj : ol.outTriangles){
        tmesh.addFace(oj.v);
    }

    for(VertexNT &v : ol.outVertices){
        VertexNC vn;
        vn.position = v.position;
        vn.normal = v.normal;
        vn.color = vec4(1,0,0,0);
        tmesh.addVertex(vn);
    }


    for(ObjTriangleGroup &tg : ol.triangleGroups){
        for(int i = 0 ; i < tg.faces ; ++i){
            ObjTriangle &face = ol.outTriangles[i+tg.startFace];
            for(int f = 0 ; f < 3 ; ++f){
                int index = face.v[f];
                tmesh.vertices[index].color = tg.material.color;
                float spec =  glm::dot(tg.material.Ks,vec3(1))/3.0f;
                tmesh.vertices[index].data.x = spec;
            }
        }
    }
}


std::shared_ptr<ColoredAsset> ObjAssetLoader::loadBasicAsset(const std::string &file, bool normalize)
{
    std::shared_ptr<ColoredAsset> asset = std::make_shared<ColoredAsset>();
    loadMeshNC(file,asset->model.mesh,normalize);
    loadBasicShaders();
    asset->create(basicAssetShader,basicAssetForwardShader,basicAssetDepthshader,basicAssetWireframeShader,normalize,false);
    return  asset;
}

std::shared_ptr<TexturedAsset> ObjAssetLoader::loadTexturedAsset(const std::string &file, bool normalize)
{
    ObjLoader2 ol(file);

    //    TexturedAsset* asset = new TexturedAsset();
    std::shared_ptr<TexturedAsset> asset = std::make_shared<TexturedAsset>();
    TriangleMesh<VertexNTD,GLuint> &tmesh = asset->model.mesh;

    for(ObjTriangle &oj : ol.outTriangles){
        tmesh.addFace(oj.v);
    }

    for(VertexNT &v : ol.outVertices){
        VertexNTD vn;
        vn.position = v.position;
        vn.normal = v.normal;
        vn.texture = v.texture;
        tmesh.addVertex(vn);
    }

    for(ObjTriangleGroup &otg : ol.triangleGroups){
        TexturedAsset::TextureGroup tg;
        tg.indices = otg.faces * 3;
        tg.startIndex = otg.startFace * 3;
        tg.texture = otg.material.map_Kd;
        if(tg.texture){
            tg.texture->setWrap(GL_REPEAT);
            tg.texture->generateMipmaps();
            asset->groups.push_back(tg);
        }
    }

    for(ObjTriangleGroup &tg : ol.triangleGroups){
        for(int i = 0 ; i < tg.faces ; ++i){
            ObjTriangle &face = ol.outTriangles[i+tg.startFace];
            for(int f = 0 ; f < 3 ; ++f){
                int index = face.v[f];
                float spec =  glm::dot(tg.material.Ks,vec3(1))/3.0f;
                tmesh.vertices[index].data.x = spec;
            }
        }
    }
    loadTextureShaders();
    asset->create(texturedAssetShader,texturedAssetForwardShader,texturedAssetDepthShader,texturedAssetWireframeShader,normalize,false);

    return asset;
}

}
