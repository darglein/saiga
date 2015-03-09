#include "rendering/object3d.h"



mat4 Object3D::getModelMatrix()
{
    mat4 model;
    model = glm::mat4_cast(rot)*glm::scale(mat4(),scale);
    model[3] = vec4(position,1);
    return model;
}

void Object3D::getModelMatrix(mat4 &model){
    model = getModelMatrix();
}

void Object3D::calculateModel(){
    model = glm::mat4_cast(rot)*glm::scale(mat4(),scale);
    model[3] = vec4(position,1);
}

void Object3D::setSimpleDirection(vec3 dir){
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

void Object3D::rotateLocal2(vec3 axis, float angle){
//    mat4 rot = glm::rotate(mat4(),degreesToRadians(angle),axis);
//    rotation =  rot*rotation ;
//    calculateModel();
}


void Object3D::rotateLocal(vec3 axis, float angle){
//    mat4 rot = glm::rotate(mat4(),degreesToRadians(angle),axis);
//    rotation =  rotation * rot ;


    this->rot = glm::rotate(this->rot,degreesToRadians(angle),axis);
//    rotation = glm::mat4_cast(this->rot);
//    calculateModel();
}


void Object3D::rotateGlobal(vec3 axis, float angle){
    axis = vec3((glm::inverse(rot)) * vec4(axis,0));
    axis = glm::normalize(axis);
//    mat4 rot = glm::rotate(mat4(),degreesToRadians(angle),axis);
//    rotation =  rotation * rot ;

    this->rot = glm::rotate(this->rot,degreesToRadians(angle),axis);
//    rotation = glm::mat4_cast(this->rot);
//    calculateModel();
//    model = glm::rotate(model,degreesToRadians(angle),axis);
}

void Object3D::translateLocal(vec4 d){
    translateLocal(vec3(d));
}

void Object3D::translateGlobal(vec4 d){
    translateGlobal(vec3(d));
}

void Object3D::translateLocal(vec3 d){
//    base = glm::translate(base,d);

    vec4 d2 = rot*vec4(d,1);
    translateGlobal(d2);
//    calculateModel();
}

void Object3D::translateGlobal(vec3 d){
    position += d;
//    calculateModel();
}

void Object3D::normalize(){
//    model[0] = glm::normalize(model[0]);
//    model[1] = glm::normalize(model[1]);
//    model[2] = glm::normalize(model[2]);
}

//void Object3D::scale(vec3 s){
//    mat4 scale = glm::scale(mat4(), s);
//    size = size *scale ;
////    calculateModel();
//}

vec3 Object3D::getScale(){
    return scale;
}

void Object3D::setScale(vec3 s){
   scale = s;
}

//void Object3D::getViewMatrix(mat4& view){
//    view= glm::mat4_cast(rot);
//    view[3] = vec4(position,1);
//    view = glm::inverse(view);
//}


