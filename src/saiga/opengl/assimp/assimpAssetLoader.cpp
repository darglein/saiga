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
    asset->create();

    return std::shared_ptr<ColoredAsset>(asset);
}

std::shared_ptr<TexturedAsset> AssimpAssetLoader::loadTexturedAsset(const std::string& file, bool normalize)
{
    SAIGA_ASSERT(0);
    return nullptr;
#    if 0
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

    asset->create();

    return std::shared_ptr<TexturedAsset>(asset);
#    endif
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

    asset->create();
    return std::shared_ptr<AnimatedAsset>(asset);
}



std::shared_ptr<Asset> AssimpAssetLoader::loadAsset(const std::string& file)
{
    return loadBasicAsset(file);
}

}  // namespace Saiga
#endif
