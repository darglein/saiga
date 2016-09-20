#include "saiga/assimp/assimpAssetLoader.h"

#include "saiga/assimp/assimpLoader.h"
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

AnimatedAsset *AssimpAssetLoader::loadAnimatedAsset(const std::string &file, bool normalize)
{
    AssimpLoader al(file);
//    al.verbose = true;
    al.printInfo();

    int meshCount = al.scene->mNumMeshes;

    if(meshCount==0){
        return nullptr;
    }

    AnimatedAsset* asset = new AnimatedAsset();

    TriangleMesh<BoneVertexCD,GLuint> &tmesh = asset->mesh;


    al.loadBones();

    for(int i=0;i<meshCount;++i){
        TriangleMesh<BoneVertexCD,GLuint> tmesh3;
        al.getPositions(i,tmesh3);
        al.getFaces(i,tmesh3);
        al.getNormals(i,tmesh3);
        al.getColors(i,tmesh3);
        al.getBones(i,tmesh3);
        al.getData(i,tmesh3);
        tmesh.addMesh(tmesh3);
    }


    for(BoneVertexCD &bv : asset->mesh.vertices){
        bv.normalizeWeights();
    }

    asset->boneCount = al.boneOffsets.size();
    asset->boneMap = al.boneMap;
    asset->nodeindexMap = al.nodeindexMap;
    asset->boneOffsets = al.boneOffsets;

    for(unsigned int i=0;i<asset->boneOffsets.size();++i){
        asset->inverseBoneOffsets.push_back(glm::inverse(asset->boneOffsets[i]));
    }

    int animationCount = al.scene->mNumAnimations;

    asset->animations.resize(animationCount);
    for(int i=0;i<animationCount;++i){
        al.getAnimation(i,0,asset->animations[i]);
    }

    for(BoneVertexCD &v : asset->mesh.vertices){
        vec3 c = v.color;
        c = Color::srgb2linearrgb(c);
        v.color = c;
    }


//    std::vector<mat4> boneMatrices(al.boneOffsets.size());
//    for(int i = 0 ; i < al.boneOffsets.size() ; ++i){
//        mat4 randomTransformation = glm::translate(mat4(),vec3(1,i,3));
//        randomTransformation = glm::rotate(randomTransformation,123.123f+i,vec3(-14,2,i));
//        randomTransformation = glm::scale(randomTransformation,vec3(i,3.5f,5.1f));
//        randomTransformation = glm::rotate(randomTransformation,123.123f*i,vec3(4,2,-5*i));
//        boneMatrices[i] = randomTransformation * al.boneOffsets[i];
//    }

//    int i = 0;
//    for(BoneVertexNC v : asset->mesh.vertices){
//        v.apply(boneMatrices);
//        if(i%10==0){
//            cout<<v.activeBones()<<" "<<v.position<<endl;
//        }
//        i++;
//    }



    //    asset->create(file,basicAssetShader,basicAssetDepthshader,basicAssetWireframeShader,normalize,false);
    asset->create(file,animatedAssetShader,animatedAssetDepthshader,animatedAssetWireframeShader,normalize,false);




    return asset;
}



Asset *AssimpAssetLoader::loadAsset(const std::string &file)
{
    return loadBasicAsset(file);
}


