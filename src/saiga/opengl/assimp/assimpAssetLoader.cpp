/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"
#if defined(SAIGA_USE_OPENGL) && defined(SAIGA_USE_ASSIMP)
#    include "saiga/opengl/assimp/assimpAssetLoader.h"
#    include "saiga/opengl/assimp/assimpLoader.h"
#    include "saiga/opengl/shader/shaderLoader.h"
#    include "saiga/opengl/texture/TextureLoader.h"

namespace Saiga
{
AssimpAssetLoader::AssimpAssetLoader() {}

AssimpAssetLoader::~AssimpAssetLoader() {}


std::shared_ptr<ColoredAsset> AssimpAssetLoader::loadBasicAsset(const std::string& file, bool normalize)
{
    AssimpLoader al(file);

    int meshCount = al.scene->mNumMeshes;

    if (meshCount == 0)
    {
        return nullptr;
    }

    ColoredAsset* asset = new ColoredAsset();

    auto& tmesh = *asset;


    for (int i = 0; i < meshCount; ++i)
    {
        TriangleMesh<VertexNC, GLuint> tmesh3;
        al.getPositions(i, tmesh3);
        al.getFaces(i, tmesh3);
        al.getNormals(i, tmesh3);
        al.getColors(i, tmesh3);
        al.getData(i, tmesh3);
        tmesh.addMesh(tmesh3);
    }

    loadBasicShaders();
    asset->create(basicAssetShader, basicAssetForwardShader, basicAssetDepthshader, basicAssetWireframeShader,
                  normalize, false);


    return std::shared_ptr<ColoredAsset>(asset);
}

std::shared_ptr<TexturedAsset> AssimpAssetLoader::loadTexturedAsset(const std::string& file, bool normalize)
{
    AssimpLoader al(file);

    al.printInfo();

    int meshCount = al.scene->mNumMeshes;

    if (meshCount == 0)
    {
        return nullptr;
    }


    TexturedAsset* asset = new TexturedAsset();

    auto& tmesh = *asset;


    for (int i = 0; i < meshCount; ++i)
    {
        TriangleMesh<VertexNT, GLuint> tmesh3;

        const aiMesh* mesh         = al.scene->mMeshes[i];
        const aiMaterial* material = al.scene->mMaterials[mesh->mMaterialIndex];
        aiString texturepath;
        material->GetTexture(aiTextureType_DIFFUSE, 0, &texturepath);


        al.getPositions(i, tmesh3);
        al.getNormals(i, tmesh3);
        al.getTextureCoordinates(i, tmesh3);

        al.getFaces(i, tmesh3);



        TexturedAsset::TextureGroup tg;
        tg.indices    = tmesh3.faces.size() * 3;
        tg.startIndex = tmesh.faces.size() * 3;
        tg.texture    = TextureLoader::instance()->load(texturepath.C_Str());

        if (tg.texture)
        {
            tg.texture->setWrap(GL_REPEAT);
            asset->groups.push_back(tg);

            tmesh.addMesh(tmesh3);
        }
    }

    loadTextureShaders();
    asset->create(texturedAssetShader, texturedAssetForwardShader, texturedAssetDepthShader,
                  texturedAssetWireframeShader, normalize, false);

    return std::shared_ptr<TexturedAsset>(asset);
}

std::shared_ptr<AnimatedAsset> AssimpAssetLoader::loadAnimatedAsset(const std::string& file, bool normalize)
{
    AssimpLoader al(file);
    //    al.verbose = true;
    al.printInfo();

    int meshCount = al.scene->mNumMeshes;

    if (meshCount == 0)
    {
        return nullptr;
    }

    AnimatedAsset* asset = new AnimatedAsset();

    auto& tmesh = *asset;


    al.loadBones();

    for (int i = 0; i < meshCount; ++i)
    {
        TriangleMesh<BoneVertexCD, GLuint> tmesh3;
        al.getPositions(i, tmesh3);
        al.getFaces(i, tmesh3);
        al.getNormals(i, tmesh3);
        al.getColors(i, tmesh3);
        al.getBones(i, tmesh3);
        al.getData(i, tmesh3);
        tmesh.addMesh(tmesh3);
    }


    for (BoneVertexCD& bv : asset->vertices)
    {
        bv.normalizeWeights();
    }

    asset->boneCount    = al.boneOffsets.size();
    asset->boneMap      = al.boneMap;
    asset->nodeindexMap = al.nodeindexMap;
    asset->boneOffsets  = al.boneOffsets;

    for (unsigned int i = 0; i < asset->boneOffsets.size(); ++i)
    {
        asset->inverseBoneOffsets.push_back(inverse(asset->boneOffsets[i]));
    }

    int animationCount = al.scene->mNumAnimations;

    asset->animations.resize(animationCount);
    for (int i = 0; i < animationCount; ++i)
    {
        al.getAnimation(i, 0, asset->animations[i]);
    }

    //    for(BoneVertexCD &v : asset.vertices){
    //        vec3 c = v.color;
    //        c = Color::srgb2linearrgb(c);
    //        v.color = c;
    //    }


    //    AlignedVector<mat4> boneMatrices(al.boneOffsets.size());
    //    for(int i = 0 ; i < al.boneOffsets.size() ; ++i){
    //        mat4 randomTransformation = translate(mat4::Identity(),vec3(1,i,3));
    //        randomTransformation = rotate(randomTransformation,123.123f+i,vec3(-14,2,i));
    //        randomTransformation = scale(randomTransformation,vec3(i,3.5f,5.1f));
    //        randomTransformation = rotate(randomTransformation,123.123f*i,vec3(4,2,-5*i));
    //        boneMatrices[i] = randomTransformation * al.boneOffsets[i];
    //    }

    //    int i = 0;
    //    for(BoneVertexNC v : asset.vertices){
    //        v.apply(boneMatrices);
    //        if(i%10==0){
    //            std::cout<<v.activeBones()<<" "<<v.position<<endl;
    //        }
    //        i++;
    //    }



    //    asset->create(file,basicAssetShader,basicAssetDepthshader,basicAssetWireframeShader,normalize,false);
    loadAnimatedShaders();
    asset->create(animatedAssetShader, animatedAssetForwardShader, animatedAssetDepthshader,
                  animatedAssetWireframeShader, normalize, false);



    return std::shared_ptr<AnimatedAsset>(asset);
}



std::shared_ptr<Asset> AssimpAssetLoader::loadAsset(const std::string& file)
{
    return loadBasicAsset(file);
}

}  // namespace Saiga
#endif
