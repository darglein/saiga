#include "saiga/rendering/object3d.h"


void Object3D::setSimpleDirection(const glm::vec3 &dir){
    glm::mat4 rotmat;
    rotmat[0] = vec4(glm::normalize(glm::cross(dir,vec3(0,1,0))),0);
    rotmat[1] = vec4(0,1,0,0);
    rotmat[2] = vec4(-dir,0);

    this->rot = glm::normalize(glm::quat(rotmat));

//    calculateModel();
}



void Object3D::turn(float angleX, float angleY){
    rotateGlobal(vec3(0,1,0),angleX);
    mat4 modeltmp = model;
    rotateLocal(vec3(1,0,0),angleY);
    if (model[1][1] < 0){
        model = modeltmp;
    }
//    calculateModel();

//    std::cout<<this->getDirection()<<" "<<(rot*vec4(0,0,1,0))<<std::endl;
//    std::cout<<this->getRightVector()<<" "<<(rot*vec4(1,0,0,0))<<std::endl;
//    std::cout<<this->getUpVector()<<" "<<(rot*vec4(0,1,0,0))<<std::endl;
}


void Object3D::rotateAroundPoint(const glm::vec3& point, const glm::vec3& axis, float angle){
    rotateLocal(axis,angle);

    translateGlobal(-point);
    quat qrot = glm::angleAxis(degreesToRadians(angle),axis);
    position = vec3(qrot*vec4(position,1));
    translateGlobal(point);
}






