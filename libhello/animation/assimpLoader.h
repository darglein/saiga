#pragma once

#include <libhello/geometry/triangle_mesh.h>

#include <libhello/animation/animation.h>

#include <type_traits>

#include <map>

#include <assimp/Importer.hpp> // C++ importer interface
#include <assimp/scene.h> // Output data structure
#include <assimp/postprocess.h> // Post processing flags
#include <assimp/cimport.h>


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




class AssimpLoader{
public:
    bool verbose = true;

    const aiScene *scene = nullptr;
    Assimp::Importer importer;

    int boneCount = 0;
    std::map<std::string,int> boneMap;
    std::vector<mat4> boneOffsets;

    int nodeCount = 0;
    std::map<std::string,AnimationNode*> nodeMap;

    AnimationNode rootNode;
public:
    AssimpLoader(){}
    AssimpLoader(const std::string &file);

    void loadBones();


    void loadFile(const std::string &file);


    template<typename vertex_t>
    void getMesh(int id, TriangleMesh<vertex_t, GLuint> &out);

    template<typename vertex_t>
    void getPositions(int id, TriangleMesh<vertex_t, GLuint> &out);

    template<typename vertex_t>
    void getNormals(int id,  TriangleMesh<vertex_t, GLuint> &out);

    template<typename vertex_t>
    void getBones(int id,  TriangleMesh<vertex_t, GLuint> &out);

    template<typename vertex_t>
    void getFaces(int id, TriangleMesh<vertex_t, GLuint> &out);

    template<typename vertex_t>
    void getColors(int id, TriangleMesh<vertex_t, GLuint> &out);

    void getAnimation(int animationId, int meshId, Animation &out);

    void transformmesh(const aiMesh *amesh, std::vector<mat4> &boneMatrices);
    void createFrames(const aiMesh *mesh, aiAnimation *anim, std::vector<AnimationFrame> &animationFrames);

    void createKeyFrames(const aiMesh *mesh, aiAnimation *anim, std::vector<AnimationFrame> &animationFrames);
    int countNodes(aiNode *node, AnimationNode &an);
    mat4 composematrix(vec3 t, quat q, vec3 s);
private:
    int animationlength(aiAnimation *anim);
    aiNode *findnode(aiNode *node, char *name);
    void transformnode(aiMatrix4x4 *result, aiNode *node);
    mat4 convert(aiMatrix4x4 mat);
    void composematrix(aiMatrix4x4 *m, aiVector3D *t, aiQuaternion *q, aiVector3D *s);
};


//type trait that checks if a member name exists in a type
#define HAS_MEMBER(_M,_NAME) \
    template <typename T>\
    class _NAME \
{\
    typedef char one;\
    typedef long two;\
    template <typename C> static one test( decltype(&C::_M) ) ;\
    template <typename C> static two test(...);\
    public:\
    enum { value = sizeof(test<T>(0)) == sizeof(char) };\
    };


HAS_MEMBER(position,has_position)
HAS_MEMBER(normal,has_normal)
HAS_MEMBER(texture,has_texture)
HAS_MEMBER(boneIndices,has_boneIndices)
HAS_MEMBER(boneWeights,has_boneWeights)

#define ENABLE_IF_FUNCTION(_NAME,_P1,_P2,_TRAIT) \
    template<class T> \
    void \
    _NAME(_P1,_P2, typename std::enable_if<_TRAIT<T>::value, T>::type* = 0)

#define ENABLED_FUNCTION(_NAME,_P1,_P2,_TRAIT) \
    ENABLE_IF_FUNCTION(_NAME,_P1,_P2,!_TRAIT){} \
    ENABLE_IF_FUNCTION(_NAME,_P1,_P2,_TRAIT)


#define ENABLE_IF_FUNCTION3(_NAME,_P1,_P2,_P3,_TRAIT) \
    template<class T> \
    void \
    _NAME(_P1,_P2,_P3, typename std::enable_if<_TRAIT<T>::value, T>::type* = 0)

#define ENABLED_FUNCTION3(_NAME,_P1,_P2,_P3,_TRAIT) \
    ENABLE_IF_FUNCTION3(_NAME,_P1,_P2,_P3,!_TRAIT){} \
    ENABLE_IF_FUNCTION3(_NAME,_P1,_P2,_P3,_TRAIT)




//these function will be executed if the type has the specified trait.
//if not nothing will be done

ENABLED_FUNCTION(loadPosition,T& vertex,const aiVector3D &v,has_position){
    vertex.position = vec3(v.x,v.y,v.z);
}


ENABLED_FUNCTION(loadNormal,T& vertex,const aiVector3D &v,has_normal){
    vertex.normal = vec3(v.x,v.y,v.z);
}


ENABLED_FUNCTION(loadTexture,T& vertex,const aiVector3D &v,has_texture){
    vertex.texture = vec2(v.x,v.y);
}


ENABLED_FUNCTION3(loadBoneWeight,T& vertex, int index, float weight,has_boneWeights){
    vertex.addBone(index,weight);
}



template<typename vertex_t>
void AssimpLoader::getMesh(int id,  TriangleMesh<vertex_t, GLuint> &out){
    const aiMesh *mesh = scene->mMeshes[id];


    out.vertices.resize(mesh->mNumVertices);

    if(mesh->HasPositions()){
        for(unsigned int i=0;i<mesh->mNumVertices;++i){
            vertex_t &bv = out.vertices[i];
            loadPosition(bv,mesh->mVertices[i]);
        }
    }

    if(mesh->HasNormals()){
        for(unsigned int i=0;i<mesh->mNumVertices;++i){
            vertex_t &bv = out.vertices[i];
            loadNormal(bv,mesh->mNormals[i]);
        }
    }

    if(mesh->HasTextureCoords(0)){
        for(unsigned int i=0;i<mesh->mNumVertices;++i){
            vertex_t &bv = out.vertices[i];
            loadTexture(bv,mesh->mTextureCoords[i][0]);
        }
    }

    if(mesh->HasFaces()){
        for(unsigned int i=0;i<mesh->mNumFaces;++i){
            aiFace* f = mesh->mFaces+i;
            if(f->mNumIndices != 3){
                cout<<"Mesh not triangulated!!!"<<endl;
                continue;
            }
            out.addFace(f->mIndices);
        }
    }

    if(mesh->HasBones()){
        for(unsigned int i=0;i<mesh->mNumBones;++i){
            aiBone* b = mesh->mBones[i];
            for(unsigned int j=0;j<b->mNumWeights;++j){
                aiVertexWeight* vw = b->mWeights+j;
                vertex_t& bv = out.vertices[vw->mVertexId];
                loadBoneWeight(bv,i,vw->mWeight);
            }
        }
    }
}


template<typename vertex_t>
void AssimpLoader::getPositions(int id,  TriangleMesh<vertex_t, GLuint> &out){
    const aiMesh *mesh = scene->mMeshes[id];

    out.vertices.resize(mesh->mNumVertices);

    if(mesh->HasPositions()){
        for(unsigned int i=0;i<mesh->mNumVertices;++i){
            vertex_t &bv = out.vertices[i];
            loadPosition(bv,mesh->mVertices[i]);


//            aiColor4D* c = mesh->mColors[i];
//            cout<<"color "<<c->r<<","<<c->g<<","<<c->b<<","<<c->a<<endl;
        }
    }

}


template<typename vertex_t>
void AssimpLoader::getNormals(int id,  TriangleMesh<vertex_t, GLuint> &out){
    const aiMesh *mesh = scene->mMeshes[id];

    out.vertices.resize(mesh->mNumVertices);

    if(mesh->HasNormals()){
        for(unsigned int i=0;i<mesh->mNumVertices;++i){
            vertex_t &bv = out.vertices[i];
            loadNormal(bv,mesh->mNormals[i]);
        }
    }

}

template<typename vertex_t>
void AssimpLoader::getBones(int id,  TriangleMesh<vertex_t, GLuint> &out){
    const aiMesh *mesh = scene->mMeshes[id];

    out.vertices.resize(mesh->mNumVertices);


    if(mesh->HasBones()){
        for(unsigned int i=0;i<mesh->mNumBones;++i){
            aiBone* b = mesh->mBones[i];
            std::string str(b->mName.data);
            if(boneMap.find(str)==boneMap.end()){
                assert(0);
            }
            int index = boneMap[str];
            for(unsigned int j=0;j<b->mNumWeights;++j){
                aiVertexWeight* vw = b->mWeights+j;
                vertex_t& bv = out.vertices[vw->mVertexId];
                loadBoneWeight(bv,index,vw->mWeight);
            }
        }
    }

}



template<typename vertex_t>
void AssimpLoader::getFaces(int id,  TriangleMesh<vertex_t, GLuint> &out){
    const aiMesh *mesh = scene->mMeshes[id];

    if(mesh->HasFaces()){
        for(unsigned int i=0;i<mesh->mNumFaces;++i){
            aiFace* f = mesh->mFaces+i;
            if(f->mNumIndices != 3){
                cout<<"Mesh not triangulated!!!"<<endl;
                continue;
            }
            out.addFace(f->mIndices);
        }
    }
}



template<typename vertex_t>
void AssimpLoader::getColors(int id,  TriangleMesh<vertex_t, GLuint> &out){
    const aiMesh *mesh = scene->mMeshes[id];

    out.vertices.resize(mesh->mNumVertices);

    aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];

    aiColor3D color (0.f,0.f,0.f);
    material->Get(AI_MATKEY_COLOR_DIFFUSE,color);


        for(unsigned int i=0;i<mesh->mNumVertices;++i){
            vertex_t &bv = out.vertices[i];

            bv.color = vec3(color.r,color.g,color.b);
        }


}


