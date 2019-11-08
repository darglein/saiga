/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "assimpLoader.h"

#if defined(SAIGA_USE_OPENGL) && defined(SAIGA_USE_ASSIMP)
#    include "saiga/core/util/assert.h"

#    include <iostream>
namespace Saiga
{
AssimpLoader::AssimpLoader(const std::string& _file) : file(_file)
{
    loadFile(file);
}

void AssimpLoader::loadFile(const std::string& _file)
{
    file = _file;
    importer.SetPropertyInteger(AI_CONFIG_PP_LBW_MAX_WEIGHTS, 4);

    int flags = aiProcess_Triangulate;
    flags |= aiProcess_JoinIdenticalVertices;
    //    flags |= aiProcess_GenSmoothNormals;
    flags |= aiProcess_GenUVCoords;
    flags |= aiProcess_TransformUVCoords;
    //    flags |= aiProcess_RemoveComponent;
    flags |= aiProcess_LimitBoneWeights;
    flags |= aiProcess_GenNormals;

    scene = importer.ReadFile(file, flags);
    // If the import failed, report it
    if (!scene)
    {
        std::cout << importer.GetErrorString() << std::endl;
        SAIGA_ASSERT(0);
    }

    if (verbose)
    {
        printInfo();
    }
}

void AssimpLoader::printInfo()
{
    std::cout << ">> AssimpLoader: " << file << " ";
    std::cout << "Animations " << scene->mNumAnimations << ", Cameras " << scene->mNumCameras << ", Lights "
              << scene->mNumLights << ", Materials " << scene->mNumMaterials << ", Meshes " << scene->mNumMeshes
              << ", Textures " << scene->mNumTextures << std::endl;

    for (unsigned int m = 0; m < scene->mNumMeshes; ++m)
    {
        const aiMesh* mesh = scene->mMeshes[m];
        std::cout << ">>> Mesh " << m << ": Material id " << mesh->mMaterialIndex << ", Vertices " << mesh->mNumVertices
                  << ", Faces " << mesh->mNumFaces << std::endl;
    }

    for (unsigned int m = 0; m < scene->mNumMaterials; ++m)
    {
        const aiMaterial* material = scene->mMaterials[m];
        printMaterialInfo(material);
    }
}

void AssimpLoader::printMaterialInfo(const aiMaterial* material)
{
    aiString texturepath;
    material->GetTexture(aiTextureType_DIFFUSE, 0, &texturepath);

    aiColor3D cd(0.f, 0.f, 0.f);
    material->Get(AI_MATKEY_COLOR_DIFFUSE, cd);

    aiColor3D ce(0.f, 0.f, 0.f);
    material->Get(AI_MATKEY_COLOR_EMISSIVE, ce);

    aiColor3D cs(0.f, 0.f, 0.f);
    material->Get(AI_MATKEY_COLOR_SPECULAR, cs);


    std::cout << ">>>> Material: "
              << "Color Diffuse (" << cd.r << " " << cd.g << " " << cd.b << "), Color Emissive (" << ce.r << " " << ce.g
              << " " << ce.b << "), Color Specular (" << cs.r << " " << cs.g << " " << cs.b << "), Diffuse texture "
              << texturepath.C_Str() << std::endl;
}

void AssimpLoader::loadBones()
{
    for (unsigned int m = 0; m < scene->mNumMeshes; ++m)
    {
        const aiMesh* mesh = scene->mMeshes[m];
        for (unsigned int i = 0; i < mesh->mNumBones; ++i)
        {
            aiBone* b = mesh->mBones[i];

            std::string str(b->mName.data);
            if (boneMap.find(str) == boneMap.end())
            {
                mat4 boneOffset = convert(b->mOffsetMatrix);
                boneOffsets.push_back(boneOffset);
                boneMap[str] = boneCount++;
            }
        }
    }

    rootNode = createNodeTree(scene->mRootNode);
    SAIGA_ASSERT(rootNode == 0);
    //    std::cout<<"unique nodes: "<<nodeCount<<endl;

    if (verbose) std::cout << ">>Created node map: " << nodeCount << " nodes, " << boneCount << " bones." << std::endl;
}

void AssimpLoader::getAnimation(int animationId, int meshId, Animation& out)
{
    (void)meshId;
    // const aiMesh *mesh = scene->mMeshes[meshId];

    //    out.boneMatrices.resize(boneCount);
    //    transformmesh(mesh,out.boneMatrices);


    aiAnimation* curanim = scene->mAnimations[animationId];

    //    createFrames(mesh,curanim,out.animationFrames);
    createKeyFrames(curanim, out.keyFrames);

    out.frameCount = out.keyFrames.size();
    out.name       = curanim->mName.data;

    // the duration is the time of the last keyframe
    out.duration = out.keyFrames.back().time;

    out.boneOffsets = boneOffsets;
    out.boneCount   = boneCount;

    auto tps = curanim->mTicksPerSecond;
    for (AnimationFrame& af : out.keyFrames)
    {
        af.time = af.time / tps;
        af.calculateBoneMatrices(out);
    }
    //    out.print();

    if (verbose) std::cout << ">>loaded animation " << out.name << ": " << out.frameCount << " frames" << std::endl;
}


void AssimpLoader::createKeyFrames(aiAnimation* anim, std::vector<AnimationFrame>& animationFrames)
{
    aiVectorKey *p0, *s0;
    aiQuatKey* r0;
    aiVector3D p, s;
    aiQuaternion r;


    // the last frame is the same as the first
    int frames = animationlength(anim);
    SAIGA_ASSERT(frames > 0);
    //    frames = 1;

    animationFrames.resize(frames);

    // assimp supports animation that have different numbers of position rotation and scaling keys.
    // this is not supported here. Every keyframe has to have exactly one of those keys.
    SAIGA_ASSERT(anim->mNumChannels > 0);
    SAIGA_ASSERT((int)anim->mChannels[0]->mNumPositionKeys == frames);
    SAIGA_ASSERT((int)anim->mChannels[0]->mNumRotationKeys == frames);
    SAIGA_ASSERT((int)anim->mChannels[0]->mNumScalingKeys == frames);

    // we shift the animation so that it starts at time 0
    double firstKeyFrameTime = anim->mChannels[0]->mPositionKeys[0].mTime;

    for (int frame = 0; frame < frames; ++frame)
    {
        for (auto& an : animationNodes)
        {
            an.reset();
            an.keyFramed = false;
        }

        double keyFrameTime = anim->mChannels[0]->mPositionKeys[frame].mTime;


        for (unsigned int i = 0; i < anim->mNumChannels; i++)
        {
            aiNodeAnim* chan = anim->mChannels[i];
            p0               = chan->mPositionKeys + frame;
            r0               = chan->mRotationKeys + frame;
            s0               = chan->mScalingKeys + frame;

            // assimp supports that the keys do not sync to eachother.
            // this is not supported here.
            SAIGA_ASSERT(keyFrameTime == p0->mTime && keyFrameTime == r0->mTime && keyFrameTime == s0->mTime);


            p = p0->mValue;
            ;
            r = r0->mValue;
            s = s0->mValue;

            std::string str(chan->mNodeName.data);
            if (nodeMap.find(str) == nodeMap.end())
            {
                SAIGA_ASSERT(0 && "nodeMap.find(str)==nodeMap.end()");
            }
            AnimationNode& an = animationNodes[nodeMap[str]];

            an.position  = vec4(p.x, p.y, p.z, 1);
            an.rotation  = quat(r.w, r.x, r.y, r.z);
            an.scaling   = vec4(s.x, s.y, s.z, 1);
            an.keyFramed = true;
        }
        // k.initTree();
        AnimationFrame& k = animationFrames[frame];
        k.nodeCount       = nodeCount;
        SAIGA_ASSERT(rootNode == 0);
        //        k.boneOffsets = boneOffsets;
        k.nodes = animationNodes;
        k.time  = animationtime_t((keyFrameTime - firstKeyFrameTime));

        // std::cout << k.nodes.size() << std::endl;
    }
}


//========================= Assimp helper functions ==================================



int AssimpLoader::animationlength(aiAnimation* anim)
{
    unsigned int i, len = 0;
    for (i = 0; i < anim->mNumChannels; i++)
    {
        struct aiNodeAnim* chan = anim->mChannels[i];
        len                     = std::max(len, chan->mNumPositionKeys);
        len                     = std::max(len, chan->mNumRotationKeys);
        len                     = std::max(len, chan->mNumScalingKeys);
    }
    return len;
}



aiNode* AssimpLoader::findnode(struct aiNode* node, char* name)
{
    if (!strcmp(name, node->mName.data)) return node;
    for (unsigned int i = 0; i < node->mNumChildren; i++)
    {
        struct aiNode* found = findnode(node->mChildren[i], name);
        if (found) return found;
    }
    return NULL;
}

int AssimpLoader::createNodeTree(struct aiNode* node)
{
    //    std::cout<<"node "<<node->mName.data<<endl;
    //    int n = 1;


    AnimationNode an;
    int nodeIndex = animationNodes.size();

    //    int index = 0;
    std::string str(node->mName.data);
    if (nodeMap.find(str) == nodeMap.end())
    {
        //        index = nodeMap.size();
        nodeMap[str] = nodeIndex;
        nodeCount++;
    }
    else
    {
        SAIGA_ASSERT(0);
    }


    if (boneMap.find(str) != boneMap.end())
    {
        an.boneIndex = boneMap[str];
    }
    else
    {
        an.boneIndex = -1;
    }
    nodeindexMap[str] = nodeIndex;
    an.index          = nodeIndex;
    an.matrix         = convert(node->mTransformation);
    an.name           = str;
    an.children.resize(node->mNumChildren);

    animationNodes.push_back(an);

    std::vector<int> children;
    for (unsigned int i = 0; i < node->mNumChildren; i++)
    {
        children.push_back(createNodeTree(node->mChildren[i]));
    }
    an.children               = children;
    animationNodes[nodeIndex] = an;

    return nodeIndex;
}

// calculate absolute transform for node to do mesh skinning
void AssimpLoader::transformnode(aiMatrix4x4* result, aiNode* node)
{
    //    std::cout<<"transform "<<node->mName.data<<endl;
    if (node->mParent)
    {
        transformnode(result, node->mParent);
        aiMultiplyMatrix4(result, &node->mTransformation);
    }
    else
    {
        *result = node->mTransformation;
    }

    //    std::string str(node->mName.data);
    //    AnimationNode* an = nodeMap[str];
    //    an->testMat = convert(*result);
}


mat4 AssimpLoader::composematrix(vec3 position, quat rotation, vec3 scaling)
{
    mat4 t = translate(position);
    mat4 r = make_mat4(rotation);
    mat4 s = scale(scaling);



    mat4 erg = t * s * r;


    return erg;
}

void AssimpLoader::composematrix(aiMatrix4x4* m, aiVector3D* t, aiQuaternion* q, aiVector3D* s)
{
    // quat to rotation matrix
    m->a1 = 1 - 2 * (q->y * q->y + q->z * q->z);
    m->a2 = 2 * (q->x * q->y - q->z * q->w);
    m->a3 = 2 * (q->x * q->z + q->y * q->w);
    m->b1 = 2 * (q->x * q->y + q->z * q->w);
    m->b2 = 1 - 2 * (q->x * q->x + q->z * q->z);
    m->b3 = 2 * (q->y * q->z - q->x * q->w);
    m->c1 = 2 * (q->x * q->z - q->y * q->w);
    m->c2 = 2 * (q->y * q->z + q->x * q->w);
    m->c3 = 1 - 2 * (q->x * q->x + q->y * q->y);

    // scale matrix
    m->a1 *= s->x;
    m->a2 *= s->x;
    m->a3 *= s->x;
    m->b1 *= s->y;
    m->b2 *= s->y;
    m->b3 *= s->y;
    m->c1 *= s->z;
    m->c2 *= s->z;
    m->c3 *= s->z;

    // set translation
    m->a4 = t->x;
    m->b4 = t->y;
    m->c4 = t->z;

    m->d1 = 0;
    m->d2 = 0;
    m->d3 = 0;
    m->d4 = 1;
}

mat4 AssimpLoader::convert(aiMatrix4x4 mat)
{
    mat4 ret;
    //    mat[0];
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            ret.col(i)[j] = mat[j][i];
        }
    }
    return ret;
}

}  // namespace Saiga

#endif
