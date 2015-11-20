#include "saiga/assets/assetLoader.h"

#include "saiga/animation/assimpLoader.h"
#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/opengl/texture/textureLoader.h"

 AssetLoader2::AssetLoader2(){
     loadDefaultShaders();
 }

 void AssetLoader2::loadDefaultShaders()
 {
     shader = ShaderLoader::instance()->load<MVPShader>("deferred_mvp_model.glsl");
     depthshader = ShaderLoader::instance()->load<MVPShader>("deferred_mvp_model_depth.glsl");

     textureshader = ShaderLoader::instance()->load<MVPTextureShader>("texturedAsset.glsl");
     texturedepthshader = ShaderLoader::instance()->load<MVPTextureShader>("texturedAsset.glsl");
 }

ColoredAsset* AssetLoader2::loadBasicAsset(const std::string &file, bool normalize){

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

    asset->create(file,shader,depthshader,normalize,false);


    return asset;
}

TexturedAsset *AssetLoader2::loadTexturedAsset(const std::string &file, bool normalize)
{
    AssimpLoader al(file);

    al.printInfo();

    int meshCount = al.scene->mNumMeshes;

    if(meshCount==0){
        return nullptr;
    }


    TexturedAsset* asset = new TexturedAsset();

    TriangleMesh<VertexNT,GLuint> &tmesh = asset->mesh;


    for(int i=0;i<meshCount;++i){
        TriangleMesh<VertexNT,GLuint> tmesh3;

        const aiMesh *mesh = al.scene->mMeshes[i];
        const aiMaterial *material = al.scene->mMaterials[mesh->mMaterialIndex];
        aiString texturepath;
        material->GetTexture(aiTextureType_DIFFUSE,0,&texturepath);


        al.getPositions(i,tmesh3);
        al.getNormals(i,tmesh3);
        al.getTextureCoordinates(i,tmesh3);

        al.getFaces(i,tmesh3);




        TexturedAsset::TextureGroup tg;
        tg.indices = tmesh3.faces.size()*3;
        tg.startIndex = tmesh.faces.size()*3;
        tg.texture = TextureLoader::instance()->load(texturepath.C_Str());

        if(tg.texture){
            tg.texture->setWrap(GL_REPEAT);
        asset->groups.push_back(tg);

        tmesh.addMesh(tmesh3);
        }
    }
    asset->create(file,textureshader,texturedepthshader,normalize,false);

    return asset;
}



Asset *AssetLoader2::loadAsset(const std::string &file)
{
    return loadBasicAsset(file);
}


