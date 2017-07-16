#include "saiga/assets/objAssetLoader.h"

#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/opengl/texture/textureLoader.h"
#include "saiga/animation/objLoader2.h"

namespace Saiga {

ObjAssetLoader::ObjAssetLoader(){

}

ObjAssetLoader::~ObjAssetLoader()
{

}


std::shared_ptr<ColoredAsset> ObjAssetLoader::loadBasicAsset(const std::string &file, bool normalize){
    ObjLoader2 ol(file);

    ColoredAsset* asset = new ColoredAsset();
    TriangleMesh<VertexNC,GLuint> &tmesh = asset->mesh;

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



    asset->create(file,basicAssetShader,basicAssetDepthshader,basicAssetWireframeShader,normalize,false);

    return  std::shared_ptr<ColoredAsset>(asset);
}

std::shared_ptr<TexturedAsset> ObjAssetLoader::loadTexturedAsset(const std::string &file, bool normalize)
{
    ObjLoader2 ol(file);

    TexturedAsset* asset = new TexturedAsset();
    TriangleMesh<VertexNTD,GLuint> &tmesh = asset->mesh;

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

    asset->create(file,texturedAssetShader,texturedAssetDepthShader,texturedAssetWireframeShader,normalize,false);

    return std::shared_ptr<TexturedAsset>(asset);
}

}
