#pragma once

#include <saiga/opengl/opengl.h>
#include <saiga/camera/camera.h>


#include <saiga/util/glm.h>


#include <array>



template<typename camera_t>
class Controllable_Camera : public camera_t{
public:
    float movementSpeed = 1;
    float movementSpeedFast = 5;

    float rotationSpeed = 1;

    enum Key{
        Forward = 0,
        Backward = 1,
        Left = 2,
        Right = 3,
        Fast = 4,
        Up =5
    };

    std::array<bool,6> keyPressed {};


    Controllable_Camera(){}
    virtual ~Controllable_Camera(){}

    void update(float delta);
    void mouseRotate(float dx, float dy);

};

template<class camera_t>
void Controllable_Camera<camera_t>::mouseRotate(float dx, float dy){
    this->turn(dx*rotationSpeed,dy*rotationSpeed);
    this->calculateModel();
    this->updateFromModel();
}

template<class camera_t>
void Controllable_Camera<camera_t>::update(float delta){
    int FORWARD =  keyPressed[Forward] - keyPressed[Backward];
    int RIGHT = keyPressed[Right] - keyPressed[Left];

    float speed;
    if (keyPressed[Fast]){
        speed = movementSpeedFast;
    } else {
        speed = movementSpeed;
    }

    vec3 trans = delta*speed*FORWARD*vec3(0,0,-1) + delta*speed*RIGHT*vec3(1,0,0);
    vec3 transg =  vec3(0,1,0) * (delta*speed*keyPressed[Up]);
    this->translateLocal(trans);
    this->translateGlobal(transg);
    this->calculateModel();
    this->updateFromModel();
}

