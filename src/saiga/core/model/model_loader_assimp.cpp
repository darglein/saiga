/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "model_loader_assimp.h"

#include "saiga/core/util/fileChecker.h"
#include "saiga/core/util/tostring.h"

#if defined(SAIGA_USE_ASSIMP)
#    include "saiga/core/util/assert.h"

#    include <iostream>
namespace Saiga
{
static vec3 convert_color(aiColor3D aiv)
{
    return vec3(aiv.r, aiv.g, aiv.b);
}

static vec4 convert_color(aiColor4D aiv)
{
    return vec4(aiv.r, aiv.g, aiv.b, aiv.a);
}

static vec3 convert_vector(aiVector3D aiv)
{
    return vec3(aiv.x, aiv.y, aiv.z);
}
static aiVector3D convert_vector(vec3 aiv)
{
    return aiVector3D(aiv.x(), aiv.y(), aiv.z());
}


AssimpLoader::AssimpLoader(const std::string& _file) : file(_file)
{
    loadFile(file);
}

void AssimpLoader::loadFile(const std::string& _file)
{
    this->file = SearchPathes::model(_file);
    if (file == "")
    {
        std::cerr << "Could not open file " << _file << std::endl;
        std::cerr << SearchPathes::model << std::endl;
        return;
    }

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
        printInfo(scene);
    }

    loadBones();
}

void AssimpLoader::printInfo(const aiScene* scene)
{
    std::cout << ">> AssimpLoader: " << file << " ";
    std::cout << "Cameras " << scene->mNumCameras << ", Lights " << scene->mNumLights << ", Materials "
              << scene->mNumMaterials << ", Meshes " << scene->mNumMeshes << std::endl;

    std::cout << "> Meshes " << scene->mNumMeshes << std::endl;
    for (unsigned int m = 0; m < scene->mNumMeshes; ++m)
    {
        const aiMesh* mesh = scene->mMeshes[m];
        std::cout << " " << m << " " << mesh->mName.C_Str() << ": Material id " << mesh->mMaterialIndex << ", Vertices "
                  << mesh->mNumVertices << ", Faces " << mesh->mNumFaces << std::endl;
    }

    std::cout << "> Animations " << scene->mNumAnimations << std::endl;
    for (unsigned int m = 0; m < scene->mNumAnimations; ++m)
    {
        const aiAnimation* anim = scene->mAnimations[m];
        std::cout << " " << m << " " << anim->mName.C_Str() << ": Channels " << anim->mNumChannels << " MeshChannels "
                  << anim->mNumMeshChannels << " duration " << anim->mDuration << " tps " << anim->mTicksPerSecond
                  << std::endl;
    }

    std::cout << "> Textures " << scene->mNumTextures << std::endl;
    for (unsigned int m = 0; m < scene->mNumTextures; ++m)
    {
        aiTexture* tex = scene->mTextures[m];
        std::cout << tex->mFilename.C_Str() << " " << tex->achFormatHint << " " << tex->mWidth << "x" << tex->mHeight
                  << std::endl;
    }
}



static UnifiedMaterial ConvertMaterial(const aiMaterial* material)
{
    UnifiedMaterial mat;

    aiString name;

    aiString texture_diffuse, texture_emissive;


    aiColor3D color_diffuse(0.f, 0.f, 0.f);
    aiColor3D color_emissive(0.f, 0.f, 0.f);
    aiColor3D color_specular(0.f, 0.f, 0.f);


    material->Get(AI_MATKEY_NAME, name);

    material->GetTexture(aiTextureType_DIFFUSE, 0, &texture_diffuse);
    material->GetTexture(aiTextureType_EMISSIVE, 0, &texture_emissive);

    if (0)
    {
        std::cout << "Texture Counts: " << material->GetTextureCount(aiTextureType_NONE) << " "
                  << material->GetTextureCount(aiTextureType_DIFFUSE) << " "
                  << material->GetTextureCount(aiTextureType_SPECULAR) << " "
                  << material->GetTextureCount(aiTextureType_AMBIENT) << " "
                  << material->GetTextureCount(aiTextureType_EMISSIVE) << " "
                  << material->GetTextureCount(aiTextureType_HEIGHT) << " "
                  << material->GetTextureCount(aiTextureType_NORMALS) << " "
                  << material->GetTextureCount(aiTextureType_SHININESS) << " "
                  << material->GetTextureCount(aiTextureType_OPACITY) << " "
                  << material->GetTextureCount(aiTextureType_DISPLACEMENT) << " "
                  << material->GetTextureCount(aiTextureType_LIGHTMAP) << " "
                  << material->GetTextureCount(aiTextureType_REFLECTION) << " "
                  << material->GetTextureCount(aiTextureType_BASE_COLOR) << " "
                  << material->GetTextureCount(aiTextureType_NORMAL_CAMERA) << " "
                  << material->GetTextureCount(aiTextureType_EMISSION_COLOR) << " "
                  << material->GetTextureCount(aiTextureType_METALNESS) << " "
                  << material->GetTextureCount(aiTextureType_DIFFUSE_ROUGHNESS) << " "
                  << material->GetTextureCount(aiTextureType_AMBIENT_OCCLUSION) << " "
                  << material->GetTextureCount(aiTextureType_UNKNOWN) << std::endl;
    }


    material->Get(AI_MATKEY_COLOR_DIFFUSE, color_diffuse);
    material->Get(AI_MATKEY_COLOR_EMISSIVE, color_emissive);
    material->Get(AI_MATKEY_COLOR_SPECULAR, color_specular);


    mat.name             = name.C_Str();
    mat.texture_diffuse  = texture_diffuse.C_Str();
    mat.texture_emissive = texture_emissive.C_Str();


    mat.color_diffuse = make_vec4(convert_color(color_diffuse), 1);
    return mat;
}

static Image LoadEmbeddedTexture(aiTexture* tex)
{
    std::cout << "Load embedded " << tex->mFilename.C_Str() << " " << tex->achFormatHint << " " << tex->mWidth << "x"
              << tex->mHeight << std::endl;

    SAIGA_ASSERT(tex->mHeight == 0);

    size_t size_bytes  = tex->mWidth;
    std::string format = tex->achFormatHint;

    ArrayView<const char> image_data((const char*)tex->pcData, size_bytes);

    Image result;
    if (!result.loadFromMemory(image_data, format))
    {
        std::cout << "unable to load image" << std::endl;
    }
    else
    {
        std::cout << result << std::endl;
    }
    return result;
}

UnifiedModel AssimpLoader::Model()
{
    UnifiedModel model;

    TraversePrintTree(scene->mRootNode);
    //    exit(0);


    // load embedded texture
    for (unsigned int m = 0; m < scene->mNumTextures; ++m)
    {
        aiTexture* tex = scene->mTextures[m];
        model.textures.push_back(LoadEmbeddedTexture(tex));
    }

    for (unsigned int m = 0; m < scene->mNumMaterials; ++m)
    {
        const aiMaterial* material = scene->mMaterials[m];

        model.materials.push_back(ConvertMaterial(material));
    }



    for (unsigned int m = 0; m < scene->mNumMeshes; ++m)
    {
        int current_vertex = 0;
        int current_face   = 0;

        const aiMesh* mesh = scene->mMeshes[m];



        UnifiedMesh unified_mesh;
        unified_mesh.material_id = mesh->mMaterialIndex;

        if (mesh->HasPositions())
        {
            for (unsigned int i = 0; i < mesh->mNumVertices; ++i)
            {
                unified_mesh.position.push_back(convert_vector(mesh->mVertices[i]));
            }
        }
        else
        {
            throw std::runtime_error("A model without position is invalid!");
        }

        if (mesh->HasNormals())
        {
            for (unsigned int i = 0; i < mesh->mNumVertices; ++i)
            {
                unified_mesh.normal.push_back(convert_vector(mesh->mNormals[i]));
            }
        }

        if (mesh->HasVertexColors(0))
        {
            SAIGA_ASSERT(!mesh->HasVertexColors(1));

            for (unsigned int i = 0; i < mesh->mNumVertices; ++i)
            {
                unified_mesh.color.push_back(convert_color(mesh->mColors[0][i]));
            }
        }

        if (mesh->HasTextureCoords(0))
        {
            for (unsigned int i = 0; i < mesh->mNumVertices; ++i)
            {
                unified_mesh.texture_coordinates.push_back(convert_vector(mesh->mTextureCoords[0][i]).head<2>());
            }
        }



        if (mesh->HasBones())
        {
            unified_mesh.bone_info.resize(unified_mesh.position.size());
            for (unsigned int i = 0; i < mesh->mNumBones; ++i)
            {
                aiBone* b = mesh->mBones[i];
                for (unsigned int j = 0; j < b->mNumWeights; ++j)
                {
                    aiVertexWeight vw = b->mWeights[j];
                    int vid           = vw.mVertexId + current_vertex;

                    auto& bi = unified_mesh.bone_info[vid];

                    bi.addBone(i, vw.mWeight);
                }
            }
        }

        if (mesh->HasFaces())
        {
            for (unsigned int i = 0; i < mesh->mNumFaces; ++i)
            {
                aiFace* f = mesh->mFaces + i;
                if (f->mNumIndices == 3)
                {
                    ivec3 f1(f->mIndices[0], f->mIndices[1], f->mIndices[2]);

                    f1 += ivec3(current_vertex, current_vertex, current_vertex);
                    unified_mesh.triangles.push_back(f1);
                    current_face++;
                }
            }
        }


        current_vertex += mesh->mNumVertices;

        model.mesh.push_back(unified_mesh);
    }


    model.animation_system.boneMap      = boneMap;
    model.animation_system.nodeindexMap = nodeindexMap;
    model.animation_system.boneOffsets  = boneOffsets;
    for (unsigned int i = 0; i < model.animation_system.boneOffsets.size(); ++i)
    {
        model.animation_system.inverseBoneOffsets.push_back(inverse(model.animation_system.boneOffsets[i]));
    }

    int animationCount = scene->mNumAnimations;

    model.animation_system.animations.resize(animationCount);
    for (int i = 0; i < animationCount; ++i)
    {
        getAnimation(i, 0, model.animation_system.animations[i]);
    }

    return model;
}

void AssimpLoader::SaveModel(const UnifiedModel& model, const std::string& file)
{
    aiScene scene;


    scene.mRootNode = new aiNode("root");


    scene.mMaterials    = new aiMaterial*[1];
    scene.mNumMaterials = 1;
    scene.mMaterials[0] = new aiMaterial();


    scene.mMeshes                    = new aiMesh*[1];
    scene.mNumMeshes                 = 1;
    scene.mMeshes[0]                 = new aiMesh();
    scene.mMeshes[0]->mMaterialIndex = 0;

    scene.mRootNode->mMeshes    = new unsigned int[1];
    scene.mRootNode->mMeshes[0] = 0;
    scene.mRootNode->mNumMeshes = 1;

    SAIGA_ASSERT(model.mesh.size() == 1);

    // SAIGA_EXIT_ERROR("todo");
    auto pMesh = scene.mMeshes[0];

    auto uni_mesh = model.mesh.front();
    if (uni_mesh.HasPosition())
    {
        pMesh->mNumVertices = uni_mesh.position.size();
        pMesh->mVertices    = new aiVector3D[pMesh->mNumVertices];
        for (int i = 0; i < pMesh->mNumVertices; ++i)
        {
            pMesh->mVertices[i] = convert_vector(uni_mesh.position[i]);
        }
    }

    if (uni_mesh.HasNormal())
    {
        SAIGA_ASSERT(uni_mesh.normal.size() == uni_mesh.NumVertices());
        pMesh->mNormals = new aiVector3D[pMesh->mNumVertices];
        for (int i = 0; i < pMesh->mNumVertices; ++i)
        {
            pMesh->mNormals[i] = convert_vector(uni_mesh.normal[i]);
        }
    }


    if (uni_mesh.HasColor())
    {
        SAIGA_ASSERT(uni_mesh.color.size() == uni_mesh.NumVertices());
        pMesh->mColors[0] = new aiColor4D[pMesh->mNumVertices];
        for (int i = 0; i < pMesh->mNumVertices; ++i)
        {
            auto c               = uni_mesh.color[i];
            pMesh->mColors[0][i] = aiColor4D(c(0), c(1), c(2), c(3));
        }
    }

    if (uni_mesh.triangles.size() > 0)
    {
        pMesh->mNumFaces = uni_mesh.triangles.size();
        pMesh->mFaces    = new aiFace[pMesh->mNumFaces];

        for (int i = 0; i < pMesh->mNumFaces; ++i)
        {
            pMesh->mFaces[i].mNumIndices = 3;
            pMesh->mFaces[i].mIndices    = new unsigned int[3];
            for (int j = 0; j < 3; ++j)
            {
                pMesh->mFaces[i].mIndices[j] = uni_mesh.triangles[i](j);
            }
        }
    }

    std::string ending = fileEnding(file);


    Assimp::ExportProperties properties;

    if (uni_mesh.triangles.empty() && uni_mesh.lines.empty())
    {
        // This is probably a point cloud
        properties.SetPropertyBool(AI_CONFIG_EXPORT_POINT_CLOUDS, true);
    }


    Assimp::Exporter exporter;
    if (exporter.Export(&scene, ending.c_str(), file.c_str(), 0, &properties) != aiReturn_SUCCESS)
    {
        std::cout << exporter.GetErrorString() << std::endl;
        throw std::runtime_error("assimp export failed");
    }
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
    if (!createKeyFrames(curanim, out.keyFrames))
    {
        return;
    }

    out.frameCount = out.keyFrames.size();
    out.name       = curanim->mName.data;

    // the duration is the time of the last keyframe
    out.duration = out.keyFrames.back().time;

    out.boneOffsets = boneOffsets;
    out.boneCount   = boneCount;

    auto tps = curanim->mTicksPerSecond;
    for (AnimationKeyframe& af : out.keyFrames)
    {
        af.time = af.time / tps;
        af.calculateBoneMatrices(out);
    }
    //    out.print();

    if (verbose) std::cout << ">>loaded animation " << out.name << ": " << out.frameCount << " frames" << std::endl;
}


bool AssimpLoader::createKeyFrames(aiAnimation* anim, std::vector<AnimationKeyframe>& animationFrames)
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

    if (anim->mChannels[0]->mNumPositionKeys != frames || (int)anim->mChannels[0]->mNumRotationKeys != frames ||
        (int)anim->mChannels[0]->mNumScalingKeys != frames)
    {
        std::cout << "skipping animation, different number of keys currently not supported :(" << std::endl;
        return false;
    }

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
        AnimationKeyframe& k = animationFrames[frame];
        k.nodeCount          = nodeCount;
        SAIGA_ASSERT(rootNode == 0);
        //        k.boneOffsets = boneOffsets;
        k.nodes = animationNodes;
        k.time  = animationtime_t((keyFrameTime - firstKeyFrameTime));

        // std::cout << k.nodes.size() << std::endl;
    }
    return true;
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
void AssimpLoader::TraversePrintTree(aiNode* current_node, int depth)
{
    for (int i = 0; i < depth; ++i)
    {
        std::cout << " ";
    }
    mat4 t = convert(current_node->mTransformation);

    std::cout << "Node " << current_node->mName.C_Str() << " Trans: " << t.col(3).transpose() << std::endl;
    for (int i = 0; i < current_node->mNumChildren; ++i)
    {
        TraversePrintTree(current_node->mChildren[i], depth + 1);
    }
}

}  // namespace Saiga

#endif
