#pragma once

#include "saiga/config.h"

#include "saiga/util/glm.h"


class SAIGA_GLOBAL Object3D{

public:
    mat4 model;


    //required for non uniform scaled rotations
    //TODO: extra class so uniform objects are faster
    quat rot;
    vec3 scale = vec3(1);
    vec3 position = vec3(0);

    mat4 getModelMatrix() const;
    void getModelMatrix(mat4& model) const;
    void calculateModel();

    vec3 getPosition() const; //returns global position
    vec4 getDirection()const; //returns looking direction
    vec4 getRightVector() const;
    vec4 getUpVector()const;

    void setSimpleDirection(const vec3& dir); //sets looking direction to dir, up to (0,1,0) and right to cross(dir,up)

    void translateLocal(const vec3& d);
    void translateGlobal(const vec3& d);
    void translateLocal(const vec4& d);
    void translateGlobal(const vec4& d);

    void rotateLocal(const vec3& axis, float angle); //rotate around local axis (this is much faster than rotateGlobal)
    void rotateGlobal(vec3 axis, float angle);
    void rotateAroundPoint(const vec3& point, const vec3& axis, float angle);

    vec3 getScale() const;
    void setScale(const vec3& s);

    static quat getSimpleDirectionQuat(const vec3 &dir);

    //todo: remove virtual methodes
    virtual ~Object3D(){}
    virtual void setPosition(const vec3& cords);
    virtual void turn(float angleX, float angleY);

    //    virtual void getViewMatrix(mat4& view); //the view matrix is the inverse model matrix


};


inline mat4 Object3D::getModelMatrix() const
{
    mat4 mod;
	mod = mat4_cast(rot)*glm::scale(mat4(), scale);
	mod[3] = vec4(position, 1);
	return mod;
}


inline void Object3D::getModelMatrix(mat4 &mod) const{
    mod = getModelMatrix();
}

inline void Object3D::calculateModel(){
    model = createTRSmatrix(position,rot,scale);
}

inline vec3 Object3D::getPosition() const {
    return position;
}

inline vec4 Object3D::getDirection() const {
    return rot*vec4(0,0,1,0);
}

inline vec4 Object3D::getRightVector() const {
    return rot*vec4(1,0,0,0);
}

inline vec4 Object3D::getUpVector() const {
    return rot*vec4(0,1,0,0);
}

inline void Object3D::setPosition(const vec3 &cords){
    position = cords;
}


inline void Object3D::translateLocal(const vec4& d){
    translateLocal(vec3(d));
}

inline void Object3D::translateGlobal(const vec4& d){
    translateGlobal(vec3(d));
}

inline void Object3D::translateLocal(const vec3 &d){
    vec4 d2 = rot*vec4(d,1);
    translateGlobal(d2);
}

inline void Object3D::translateGlobal(const vec3& d){
    position += d;
}


inline void Object3D::rotateLocal(const vec3& axis, float angle){
    this->rot = glm::rotate(this->rot,glm::radians(angle),axis);
}


inline void Object3D::rotateGlobal(vec3 axis, float angle){
    axis = vec3((glm::inverse(rot)) * vec4(axis,0));
    axis = glm::normalize(axis);
    this->rot = glm::rotate(this->rot,glm::radians(angle),axis);
}

inline vec3 Object3D::getScale() const{
    return scale;
}

inline void Object3D::setScale(const vec3& s){
    scale = s;
}

