/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

#if defined(SAIGA_USE_OPENGL) && defined(SAIGA_USE_ASSIMP)

#    include "saiga/core/geometry/triangle_mesh.h"
#    include "saiga/core/model/UnifiedModel.h"
#    include "saiga/core/model/animation.h"

#    include <assimp/Exporter.hpp>
#    include <assimp/Importer.hpp>  // C++ importer interface
#    include <assimp/cimport.h>
#    include <assimp/postprocess.h>  // Post processing flags
#    include <assimp/scene.h>        // Output data structure
#    include <map>

#    include <type_traits>

namespace Saiga
{
class SAIGA_CORE_API AssimpLoader
{
   public:
    std::string file;
    bool verbose = false;

    const aiScene* scene = nullptr;
    Assimp::Importer importer;

    int boneCount = 0;
    std::map<std::string, int> boneMap;
    AlignedVector<mat4> boneOffsets;

    int nodeCount = 0;
    std::map<std::string, int> nodeMap;
    std::map<std::string, int> nodeindexMap;
    std::vector<AnimationNode> animationNodes;
    int rootNode = 0;

   public:
    AssimpLoader() {}
    AssimpLoader(const std::string& file);

    void printInfo(const aiScene* scene);
    void loadBones();


    void loadFile(const std::string& _file);



    UnifiedModel Model();

    void SaveModel(const UnifiedModel& model, const std::string& file);


    void getAnimation(int animationId, int meshId, Animation& out);

    void transformmesh(const aiMesh* amesh, AlignedVector<mat4>& boneMatrices);
    void createFrames(const aiMesh* mesh, aiAnimation* anim, std::vector<AnimationKeyframe>& animationFrames);

    bool createKeyFrames(aiAnimation* anim, std::vector<AnimationKeyframe>& animationFrames);
    int createNodeTree(aiNode* node);
    mat4 composematrix(vec3 t, quat q, vec3 s);

   private:
    int animationlength(aiAnimation* anim);
    aiNode* findnode(aiNode* node, char* name);
    void transformnode(aiMatrix4x4* result, aiNode* node);
    mat4 convert(aiMatrix4x4 mat);
    void composematrix(aiMatrix4x4* m, aiVector3D* t, aiQuaternion* q, aiVector3D* s);

    void TraversePrintTree(aiNode* current_node, int depth = 0);
};

}  // namespace Saiga

#endif
