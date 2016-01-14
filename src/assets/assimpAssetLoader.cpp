#include "saiga/assets/assimpAssetLoader.h"

#include "saiga/animation/assimpLoader.h"
#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/opengl/texture/textureLoader.h"

AssimpAssetLoader::AssimpAssetLoader(){

}

AssimpAssetLoader::~AssimpAssetLoader()
{

}


ColoredAsset* AssimpAssetLoader::loadBasicAsset(const std::string &file, bool normalize){

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

TexturedAsset *AssimpAssetLoader::loadTexturedAsset(const std::string &file, bool normalize)
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
    asset->create(file,texturedAssetShader,texturedAssetDepthShader,texturedAssetWireframeShader,normalize,false);

    return asset;
}



Asset *AssimpAssetLoader::loadAsset(const std::string &file)
{
    return loadBasicAsset(file);
}


