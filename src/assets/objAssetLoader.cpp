#include "saiga/assets/objAssetLoader.h"

#include "saiga/animation/assimpLoader.h"
#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/opengl/texture/textureLoader.h"

ObjAssetLoader::ObjAssetLoader(){

}

ObjAssetLoader::~ObjAssetLoader()
{

}


ColoredAsset* ObjAssetLoader::loadBasicAsset(const std::string &file, bool normalize){

    AssimpLoader al(file);

    int meshCount = al.scene->mNumMeshes;

    if(meshCount==0){
        return nullptr;
    }

    ColoredAsset* asset = new ColoredAsset();

    TriangleMesh<VertexNC,GLuint> &tmesh = asset->mesh;


    for(int i=0;i<meshCount;++i){
        TriangleMesh<VertexNC,GLuint> tmesh3;
        al.getPositions(i,tmesh3);
        al.getFaces(i,tmesh3);
        al.getNormals(i,tmesh3);
        al.getColors(i,tmesh3);
        al.getData(i,tmesh3);
        tmesh.addMesh(tmesh3);
    }

    asset->create(file,basicAssetShader,basicAssetDepthshader,basicAssetWireframeShader,normalize,false);


    return asset;
}



