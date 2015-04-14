#pragma once

#include "libhello/util/glm.h"


//a moving object is an object that moves every game tick and collides with the world

class Object3D{
private:

public:
    mat4 model;


    //required for non uniform scaled rotations
    //TODO: extra class so uniform objects are faster
    glm::quat rot;
//    mat4 rotation;
//    mat4 size;
    vec3 scale = vec3(1);
    vec3 position = vec3(0);
//    mat4 base;

    mat4 getModelMatrix() const;
    void getModelMatrix(mat4& model) const;
    void calculateModel();

    vec3 getPosition() const {return position;} //returns global position
    vec4 getDirection()const {return rot*vec4(0,0,1,0);} //returns looking direction
    vec4 getRightVector() const {return rot*vec4(1,0,0,0);}
    vec4 getUpVector()const {return rot*vec4(0,1,0,0);}

    void setSimpleDirection(const vec3& dir); //sets looking direction to dir, up to (0,1,0) and right to cross(dir,up)

    void translateLocal(const vec3& d);
    void translateGlobal(const vec3& d);
    void translateLocal(const vec4& d);
    void translateGlobal(const vec4& d);

    void rotateLocal2(const vec3& axis, float angle);
    void rotateLocal(const vec3& axis, float angle); //rotate around local axis (this is much faster than rotateGlobal)
    void rotateGlobal(glm::vec3 axis, float angle);
    void rotateAroundPoint(const vec3& point, const vec3& axis, float angle);


//    void scale(vec3 s);
    vec3 getScale() const;
    void setScale(const glm::vec3& s);
    void normalize();

    virtual void setPosition(const glm::vec3& cords){position = cords;}
    virtual void turn(float angleX, float angleY);

//    virtual void getViewMatrix(mat4& view); //the view matrix is the inverse model matrix


};


