#include "animation/animationFrame.h"

#include <glm/gtx/quaternion.hpp>
#include <iostream>

using std::cout;
using std::endl;

void AnimationFrame::setBoneDeformation(std::vector<mat4> &boneMatrices)
{
    bones = boneMatrices.size();

    this->boneMatrices.resize(bones);
    this->boneRotations.resize(bones);
    this->bonePositions.resize(bones);
    this->boneScalings.resize(bones);

//    for(int m =0;m<bones;++m){
//        mat4 &mat = boneMatrices[m];
//        this->boneMatrices[m] = boneMatrices[m];


//        bonePositions[m] = vec3(mat[3]);

//        glm::mat3 test = glm::mat3(mat);

////        mat = glm::transpose(mat);
//        boneScalings[m].x = glm::length(vec3(mat[0]));
//        boneScalings[m].y = glm::length(vec3(mat[1]));
//        boneScalings[m].z = glm::length(vec3(mat[2]));
////       mat = glm::transpose(mat);


//        test[0] /= boneScalings[m].x;
//        test[1] /= boneScalings[m].y;
//        test[2] /= boneScalings[m].z;

////        cout<<"test "<<glm::length(test[0])<<endl;
////        cout<<"test "<<test[1]<<endl;
////        cout<<"test "<<test[2]<<endl;

//        boneRotations[m] = glm::normalize(glm::quat(glm::mat3(test)));


////        boneScalings[m] = vec3(1);
//    }


    for(int m =0;m<bones;++m){
        glm::dmat4 mat(boneMatrices[m]);
        this->boneMatrices[m] = boneMatrices[m];


        bonePositions[m] = vec3(mat[3]);

        glm::dmat3 test = glm::dmat3(mat);

        mat = glm::transpose(mat);
        boneScalings[m].x = glm::length(glm::dvec3(mat[0]));
        boneScalings[m].y = glm::length(glm::dvec3(mat[1]));
        boneScalings[m].z = glm::length(glm::dvec3(mat[2]));
       mat = glm::transpose(mat);


        test[0] /= boneScalings[m].x;
        test[1] /= boneScalings[m].y;
        test[2] /= boneScalings[m].z;

//        cout<<"test "<<glm::length(test[0])<<endl;
//        cout<<"test "<<test[1]<<endl;
//        cout<<"test "<<test[2]<<endl;

//        boneRotations[m] = glm::normalize(glm::dquat(test));
        boneRotations[m] = glm::toQuat(test);

        boneRotations[m] = glm::normalize(boneRotations[m]);
//        boneScalings[m] = vec3(1);
    }


    //test calculate transformation matrcies

    for(int m =0;m<bones;++m){
        glm::dmat4 t = glm::translate(glm::dmat4(),bonePositions[m]);
        glm::dmat4 r = glm::mat4_cast(boneRotations[m]);
        glm::dmat4 s = glm::scale(glm::dmat4(),boneScalings[m]);

        glm::dmat4 newmat = t * r * s;

//        cout<<"pos "<<this->boneMatrices[m][3]<<" "<<newmat[3]<<endl;
//        cout<<"scale "<<boneScalings[m]<<endl;
//        cout<<boneMatrices[m]<<endl;
//        cout<<mat4(newmat)<<endl;
//        cout<<"-------------------------"<<endl;

//        cout<<"pos "<<bonePositions[m]<<" "<<vec3(newmat[3])<<endl;

//        glm::mat3 test = glm::mat3(newmat);
//        test[0] /= boneScalings[m].x;
//        test[1] /= boneScalings[m].y;
//        test[2] /= boneScalings[m].z;
//        quat t1 = glm::normalize(glm::quat(glm::mat3(test)));


//        cout<<"rot "<<boneRotations[m]<<" "<<t1<<endl;

//        cout<<"-------------------------"<<endl;

        this->boneMatrices[m] = newmat;
    }

//    test(boneMatrices[0]);
//    test(boneMatrices[12]);
//    testd(glm::dmat4(boneMatrices[12]));
}

void AnimationFrame::test(glm::mat4 mat)
{

    //translation
    glm::vec3 trans = glm::vec3(mat[3]);

    //scaling
    glm::vec3 scaling;
    scaling.x = glm::length(glm::vec3(mat[0]));
    scaling.y = glm::length(glm::vec3(mat[1]));
    scaling.z = glm::length(glm::vec3(mat[2]));

    //rotation
    glm::mat3 inner = glm::mat3(mat);
    inner[0] = glm::normalize(inner[0]);
    inner[1] = glm::normalize(inner[1]);
    inner[2] = glm::normalize(inner[2]);

    glm::quat rot(inner);


    //convert back to matrices

    glm::mat4 t = glm::translate(glm::mat4(),trans);
    glm::mat4 r = glm::mat4_cast(rot);
    glm::mat4 s = glm::scale(glm::mat4(),scaling);

    glm::mat4 erg = t*r*s;

    cout<<mat<<endl;
    cout<<erg<<endl;

}


void AnimationFrame::testd(glm::dmat4 mat)
{
    glm::dvec3 position = glm::dvec3(mat[3]);

    glm::dmat3 test = glm::dmat3(mat);


        mat = glm::transpose(mat);
    glm::dvec3 scaling;
    scaling.x = glm::length(glm::dvec3(mat[0]));
    scaling.y = glm::length(glm::dvec3(mat[1]));
    scaling.z = glm::length(glm::dvec3(mat[2]));
     mat = glm::transpose(mat);


    test[0] = glm::normalize(test[0]);
    test[1] = glm::normalize(test[1]);
    test[2] = glm::normalize(test[2]);


    glm::dquat rot(test);





    glm::dmat4 t = glm::translate(glm::dmat4(),position);
    glm::dmat4 r = glm::mat4_cast(rot);
    glm::dmat4 s = glm::scale(glm::dmat4(),scaling);

    glm::dmat4 erg = t*r*s;

    cout<<"scale "<<scaling<<endl;
    cout<<mat<<endl;
    cout<<erg<<endl;


}

void AnimationFrame::interpolate(AnimationFrame &k0, AnimationFrame &k1, float alpha, std::vector<mat4> &out_boneMatrices)
{
    for(int i=0;i<k0.bones;++i){

//        glm::quat rot = glm::mix(k0.boneRotations[i],k1.boneRotations[i],alpha);
        glm::dquat rot = glm::slerp(k0.boneRotations[i],k1.boneRotations[i],(double)alpha);
//        rot = glm::normalize(rot);
         glm::dvec3 scale = glm::mix(k0.boneScalings[i],k1.boneScalings[i],(double)alpha);
         glm::dvec3 pos = glm::mix(k0.bonePositions[i],k1.bonePositions[i],(double)alpha);



        out_boneMatrices[i] = glm::translate( glm::dmat4(),pos)*glm::mat4_cast(rot)*glm::scale( glm::dmat4(),scale);

    }
}
