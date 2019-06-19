/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

#if defined(SAIGA_USE_OPENGL) && defined(SAIGA_USE_ASSIMP)

#include <assimp/Importer.hpp>  // C++ importer interface
#include <assimp/cimport.h>
#include <assimp/postprocess.h>  // Post processing flags
#include <assimp/scene.h>        // Output data structure
#include <map>
#include <saiga/opengl/animation/animation.h>

#include <saiga/core/geometry/triangle_mesh.h>
#include <type_traits>

namespace Saiga
{
/**
 * @brief The AssimpLoader class
 *
 * Warning: Use of SFINAE!
 *
 * Concept:
 * SFINAE test are performed to test if the templated vertex type has specific attributes.
 * For example:
 * If the vertex has a member 'normal' we want to read the normal from assimp and save them in that vertex.
 *
 */



class SAIGA_OPENGL_API AssimpLoader
{
   public:
    std::string file;
    bool verbose = false;

    const aiScene* scene = nullptr;
    Assimp::Importer importer;

    int boneCount = 0;
    std::map<std::string, int> boneMap;
    std::vector<mat4> boneOffsets;

    int nodeCount = 0;
    std::map<std::string, int> nodeMap;
    std::map<std::string, int> nodeindexMap;
    std::vector<AnimationNode> animationNodes;
    int rootNode = 0;

   public:
    AssimpLoader() {}
    AssimpLoader(const std::string& file);

    void printInfo();
    void printMaterialInfo(const aiMaterial* material);
    void loadBones();


    void loadFile(const std::string& _file);


    template <typename vertex_t>
    void getMesh(int id, TriangleMesh<vertex_t, uint32_t>& out);

    template <typename vertex_t>
    void getPositions(int id, TriangleMesh<vertex_t, uint32_t>& out);

    template <typename vertex_t>
    void getNormals(int id, TriangleMesh<vertex_t, uint32_t>& out);

    template <typename vertex_t, typename index_t>
    void getTextureCoordinates(int id, TriangleMesh<vertex_t, index_t>& out);

    template <typename vertex_t>
    void getBones(int id, TriangleMesh<vertex_t, uint32_t>& out);

    template <typename vertex_t>
    void getFaces(int id, TriangleMesh<vertex_t, uint32_t>& out);

    template <typename vertex_t>
    void getColors(int id, TriangleMesh<vertex_t, uint32_t>& out);

    template <typename vertex_t>
    void getData(int id, TriangleMesh<vertex_t, uint32_t>& out);

    void getAnimation(int animationId, int meshId, Animation& out);

    void transformmesh(const aiMesh* amesh, std::vector<mat4>& boneMatrices);
    void createFrames(const aiMesh* mesh, aiAnimation* anim, std::vector<AnimationFrame>& animationFrames);

    void createKeyFrames(aiAnimation* anim, std::vector<AnimationFrame>& animationFrames);
    int createNodeTree(aiNode* node);
    mat4 composematrix(vec3 t, quat q, vec3 s);

   private:
    int animationlength(aiAnimation* anim);
    aiNode* findnode(aiNode* node, char* name);
    void transformnode(aiMatrix4x4* result, aiNode* node);
    mat4 convert(aiMatrix4x4 mat);
    void composematrix(aiMatrix4x4* m, aiVector3D* t, aiQuaternion* q, aiVector3D* s);
};


// type trait that checks if a member name exists in a type
#define HAS_MEMBER(_M, _NAME)                          \
    template <typename T>                              \
    class _NAME                                        \
    {                                                  \
        typedef char one;                              \
        typedef long two;                              \
        template <typename C>                          \
        static one test(decltype(&C::_M));             \
        template <typename C>                          \
        static two test(...);                          \
                                                       \
       public:                                         \
        enum                                           \
        {                                              \
            value = sizeof(test<T>(0)) == sizeof(char) \
        };                                             \
    };


HAS_MEMBER(position, has_position)
HAS_MEMBER(normal, has_normal)
HAS_MEMBER(texture, has_texture)
HAS_MEMBER(boneIndices, has_boneIndices)
HAS_MEMBER(boneWeights, has_boneWeights)

#define ENABLE_IF_FUNCTION(_NAME, _P1, _P2, _TRAIT) \
    template <class T>                              \
    void _NAME(_P1, _P2, typename std::enable_if<_TRAIT<T>::value, T>::type* = 0)

#define ENABLED_FUNCTION(_NAME, _P1, _P2, _TRAIT)   \
    ENABLE_IF_FUNCTION(_NAME, _P1, _P2, !_TRAIT) {} \
    ENABLE_IF_FUNCTION(_NAME, _P1, _P2, _TRAIT)


#define ENABLE_IF_FUNCTION3(_NAME, _P1, _P2, _P3, _TRAIT) \
    template <class T>                                    \
    void _NAME(_P1, _P2, _P3, typename std::enable_if<_TRAIT<T>::value, T>::type* = 0)

#define ENABLED_FUNCTION3(_NAME, _P1, _P2, _P3, _TRAIT)   \
    ENABLE_IF_FUNCTION3(_NAME, _P1, _P2, _P3, !_TRAIT) {} \
    ENABLE_IF_FUNCTION3(_NAME, _P1, _P2, _P3, _TRAIT)



// these function will be executed if the type has the specified trait.
// if not nothing will be done

ENABLED_FUNCTION(loadPosition, T& vertex, const aiVector3D& v, has_position)
{
    vertex.position = vec4(v.x, v.y, v.z, 1);
}


ENABLED_FUNCTION(loadNormal, T& vertex, const aiVector3D& v, has_normal)
{
    vertex.normal = vec4(v.x, v.y, v.z, 0);
}


ENABLED_FUNCTION(loadTexture, T& vertex, const aiVector3D& v, has_texture)
{
    vertex.texture = vec2(v.x, v.y);
}


ENABLED_FUNCTION3(loadBoneWeight, T& vertex, int index, float weight, has_boneWeights)
{
    vertex.addBone(index, weight);
}



template <typename vertex_t>
void AssimpLoader::getMesh(int id, TriangleMesh<vertex_t, uint32_t>& out)
{
    const aiMesh* mesh = scene->mMeshes[id];


    out.vertices.resize(mesh->mNumVertices);

    if (mesh->HasPositions())
    {
        for (unsigned int i = 0; i < mesh->mNumVertices; ++i)
        {
            vertex_t& bv = out.vertices[i];
            loadPosition(bv, mesh->mVertices[i]);
        }
    }

    if (mesh->HasNormals())
    {
        for (unsigned int i = 0; i < mesh->mNumVertices; ++i)
        {
            vertex_t& bv = out.vertices[i];
            loadNormal(bv, mesh->mNormals[i]);
        }
    }

    if (mesh->HasTextureCoords(0))
    {
        for (unsigned int i = 0; i < mesh->mNumVertices; ++i)
        {
            vertex_t& bv = out.vertices[i];
            loadTexture(bv, mesh->mTextureCoords[0][i]);
        }
    }

    if (mesh->HasFaces())
    {
        for (unsigned int i = 0; i < mesh->mNumFaces; ++i)
        {
            aiFace* f = mesh->mFaces + i;
            if (f->mNumIndices != 3)
            {
                //                std::cout<<"Mesh not triangulated!!!"<<endl;
                continue;
            }
            out.addFace(f->mIndices);
        }
    }

    if (mesh->HasBones())
    {
        for (unsigned int i = 0; i < mesh->mNumBones; ++i)
        {
            aiBone* b = mesh->mBones[i];
            for (unsigned int j = 0; j < b->mNumWeights; ++j)
            {
                aiVertexWeight* vw = b->mWeights + j;
                vertex_t& bv       = out.vertices[vw->mVertexId];
                loadBoneWeight(bv, i, vw->mWeight);
            }
        }
    }
}


template <typename vertex_t>
void AssimpLoader::getPositions(int id, TriangleMesh<vertex_t, uint32_t>& out)
{
    const aiMesh* mesh = scene->mMeshes[id];

    out.vertices.resize(mesh->mNumVertices);


    if (mesh->HasPositions())
    {
        for (unsigned int i = 0; i < mesh->mNumVertices; ++i)
        {
            vertex_t& bv = out.vertices[i];
            loadPosition(bv, mesh->mVertices[i]);


            //            aiColor4D* c = mesh->mColors[i];
            //            std::cout<<"color "<<c->r<<","<<c->g<<","<<c->b<<","<<c->a<<endl;
        }
    }
}


template <typename vertex_t>
void AssimpLoader::getNormals(int id, TriangleMesh<vertex_t, uint32_t>& out)
{
    const aiMesh* mesh = scene->mMeshes[id];

    out.vertices.resize(mesh->mNumVertices);

    if (mesh->HasNormals())
    {
        for (unsigned int i = 0; i < mesh->mNumVertices; ++i)
        {
            vertex_t& bv = out.vertices[i];
            loadNormal(bv, mesh->mNormals[i]);
        }
    }
}

template <typename vertex_t, typename index_t>
void AssimpLoader::getTextureCoordinates(int id, TriangleMesh<vertex_t, index_t>& out)
{
    const aiMesh* mesh = scene->mMeshes[id];

    out.vertices.resize(mesh->mNumVertices);

    if (mesh->HasTextureCoords(0))
    {
        for (unsigned int i = 0; i < mesh->mNumVertices; ++i)
        {
            vertex_t& bv = out.vertices[i];
            //            loadTexture(bv,mesh->mTextureCoords[i][0]);
            loadTexture(bv, mesh->mTextureCoords[0][i]);
        }
    }
}

template <typename vertex_t>
void AssimpLoader::getBones(int id, TriangleMesh<vertex_t, uint32_t>& out)
{
    const aiMesh* mesh = scene->mMeshes[id];

    out.vertices.resize(mesh->mNumVertices);


    if (mesh->HasBones())
    {
        for (unsigned int i = 0; i < mesh->mNumBones; ++i)
        {
            aiBone* b = mesh->mBones[i];
            std::string str(b->mName.data);
            if (boneMap.find(str) == boneMap.end())
            {
                SAIGA_ASSERT(0);
            }
            int index = boneMap[str];
            for (unsigned int j = 0; j < b->mNumWeights; ++j)
            {
                aiVertexWeight* vw = b->mWeights + j;
                vertex_t& bv       = out.vertices[vw->mVertexId];
                loadBoneWeight(bv, index, vw->mWeight);
            }
        }
    }
}



template <typename vertex_t>
void AssimpLoader::getFaces(int id, TriangleMesh<vertex_t, uint32_t>& out)
{
    const aiMesh* mesh = scene->mMeshes[id];

    if (mesh->HasFaces())
    {
        for (unsigned int i = 0; i < mesh->mNumFaces; ++i)
        {
            aiFace* f = mesh->mFaces + i;
            if (f->mNumIndices != 3)
            {
                //                std::cout<<"Mesh not triangulated! (found face with "<<f->mNumIndices << " vertices)"
                //                <<endl;
                continue;
            }
            out.addFace(f->mIndices);
        }
    }
}



template <typename vertex_t>
void AssimpLoader::getColors(int id, TriangleMesh<vertex_t, uint32_t>& out)
{
    const aiMesh* mesh = scene->mMeshes[id];

    out.vertices.resize(mesh->mNumVertices);

    aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];

    aiColor3D color(0.f, 0.f, 0.f);
    material->Get(AI_MATKEY_COLOR_DIFFUSE, color);


    for (unsigned int i = 0; i < mesh->mNumVertices; ++i)
    {
        vertex_t& bv = out.vertices[i];

        bv.color = vec4(color.r, color.g, color.b, 0);
    }
}

template <typename vertex_t>
void AssimpLoader::getData(int id, TriangleMesh<vertex_t, uint32_t>& out)
{
    const aiMesh* mesh = scene->mMeshes[id];

    out.vertices.resize(mesh->mNumVertices);

    aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];

    aiColor3D emissivec(0.f, 0.f, 0.f);
    aiReturn emissiveret = material->Get(AI_MATKEY_COLOR_EMISSIVE, emissivec);
    float emissive       = (emissiveret == aiReturn_SUCCESS) ? emissivec.r : 0.0f;

    // if term useless, because assimp uses default values if no specular specified
    aiColor3D specularc(0.f, 0.f, 0.f);
    aiReturn specularret = material->Get(AI_MATKEY_COLOR_SPECULAR, specularc);
    float specular       = (specularret == aiReturn_SUCCESS) ? specularc.r : 0.0f;


    // assimps default specular is 0.4
    if (specular == 0.4f) specular = 0.0f;



    for (unsigned int i = 0; i < mesh->mNumVertices; ++i)
    {
        vertex_t& bv = out.vertices[i];

        bv.data = vec4(specular, emissive, 0, 0);
    }
}

}  // namespace Saiga

#endif