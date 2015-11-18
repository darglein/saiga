#include "saiga/assets/assetLoader.h"

#include "saiga/animation/assimpLoader.h"
#include "saiga/opengl/shader/shaderLoader.h"

 AssetLoader2::AssetLoader2(){

 }

ColoredAsset* AssetLoader2::loadBasicAsset(const std::string &file, bool normalize){


    ColoredAsset* asset = new ColoredAsset();
    asset->name = file;


    AssimpLoader al(file);

    int meshCount = al.scene->mNumMeshes;

    if(meshCount==0){
        return nullptr;
    }

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

    //convert blender coordinate system to our system
    mat4 m(1,0,0,0,
           0,0,-1,0,
           0,1,0,0,
           0,0,0,1);
    tmesh.transform(m);
    tmesh.transformNormal(m);



    aabb bb = tmesh.calculateAabb();

    if(normalize){
        vec3 bbmid = bb.getPosition();
        mat4 t = glm::translate(mat4(),-bbmid);
        tmesh.transform(t);
        bb.setPosition(vec3(0));
        //TODO
//        asset->rboffset = bbmid;
    }


    tmesh.createBuffers(asset->buffer);



    asset->boundingBox = bb;
//    asset->radius = glm::length(bb.getHalfExtends());

    asset->shader = ShaderLoader::instance()->load<MVPShader>("deferred_mvp_model.glsl");
    asset->depthshader = ShaderLoader::instance()->load<MVPShader>("deferred_mvp_model_depth.glsl");

    return asset;
}



Asset *AssetLoader2::loadAsset(const std::string &file)
{
    return loadBasicAsset(file);
}


